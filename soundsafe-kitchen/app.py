# ============================================================
# SoundSafe Kitchen — Streamlit Application
# app.py
# ============================================================
# HOW TO RUN:
#   Open terminal in this folder and run:
#   streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import os
import tempfile
import plotly.graph_objects as go
import pandas as pd
from scipy import signal as scipy_signal

# Path to the Streamlit app folder
streamlit_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="SoundSafe Kitchen",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colours ───────────────────────────────────────────────────
COLOR_SAFE    = "#2D6A4F"
COLOR_CAUTION = "#E8A838"
COLOR_HAZARD  = "#C94B4B"
THRESHOLD     = 0.5

# ── Audio settings (must match training) ─────────────────────
SR         = 16000
N_MELS     = 64
HOP_LENGTH = 512
N_FFT      = 2048
CLIP_DUR   = 5.0

# ============================================================
# MODEL ARCHITECTURE — must match training exactly
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.attention_fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        # x shape: (batch, channels, time)
        x_p = x.permute(0, 2, 1)           # (batch, time, channels)
        w   = torch.softmax(
            self.attention_fc(x_p), dim=1)  # (batch, time, 1)
        return (x_p * w).sum(dim=1), w      # (batch, channels)


class SoundSafeCNN(nn.Module):
    def __init__(self, n_mels=64, filters=(64, 128, 64), dropout=0.3):
        super().__init__()
        f1, f2, f3 = filters

        # Three 1D convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_mels, f1, kernel_size=5, padding=2),
            nn.BatchNorm1d(f1), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm1d(f2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm1d(f3), nn.ReLU(), nn.Dropout(dropout)
        )

        # Temporal attention layer
        self.attention = TemporalAttention(f3)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(f3, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, 64, 157)
        out = self.conv_block1(x)          # (batch, 64, 157)
        out = self.conv_block2(out)        # (batch, 128, 157)
        out = self.conv_block3(out)        # (batch, 64, 157)
        ctx, attn = self.attention(out)    # (batch, 64)
        out = self.dropout(ctx)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1), attn

    def predict_proba(self, x):
        with torch.no_grad():
            logit, attn = self.forward(x)
            return torch.sigmoid(logit), attn


# ============================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================
def preprocess_audio(audio, sr_in):
    """Resample, mono, filter, pad/trim, normalise."""
    # Resample if needed
    if sr_in != SR:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=SR)
    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    # Remove DC offset
    audio = (audio - np.mean(audio)).astype(np.float32)
    # High-pass filter at 80 Hz — removes electrical hum
    nyq = SR / 2.0
    sos = scipy_signal.butter(
        4, 80 / nyq, btype="high", output="sos")
    audio = scipy_signal.sosfiltfilt(sos, audio).astype(np.float32)
    # Fix length to exactly CLIP_DUR seconds
    target = int(CLIP_DUR * SR)
    if len(audio) > target:
        audio = audio[:target]
    elif len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    # Normalise loudness to -14 LUFS
    rms   = np.sqrt(np.mean(audio ** 2) + 1e-10)
    audio = audio * (10 ** ((-14.0 - 20 * np.log10(rms)) / 20))
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def audio_to_mel(audio):
    """Convert preprocessed audio to mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)  # Shape: (64, ~157)


def analyse_audio(audio_full, sr_in, model):
    """
    Analyse full audio in 5-second overlapping windows.
    Returns list of results per window.
    """
    # Resample and mono convert full audio
    if sr_in != SR:
        audio_full = librosa.resample(
            audio_full, orig_sr=sr_in, target_sr=SR)
    if audio_full.ndim > 1:
        audio_full = audio_full.mean(axis=0)

    window = int(CLIP_DUR * SR)        # 80,000 samples
    hop    = int(CLIP_DUR * SR / 2)    # 40,000 samples (50% overlap)

    # Make sure we have at least one window
    positions = list(range(0, max(len(audio_full) - window + 1, 1), hop))
    if not positions:
        positions = [0]

    results = []
    for start in positions:
        clip = audio_full[start:start + window]
        proc = preprocess_audio(clip, SR)
        mel  = audio_to_mel(proc)

        # Add batch dimension: (1, 64, 157)
        mel_t = torch.FloatTensor(mel).unsqueeze(0)
        prob, attn = model.predict_proba(mel_t)
        p = float(prob.item())

        # Classify risk level
        if p >= THRESHOLD:
            risk = "HAZARD"
        elif p >= 0.30:
            risk = "Caution"
        else:
            risk = "Safe"

        results.append({
            "time_sec"   : start / SR,
            "probability": p,
            "risk_level" : risk,
            "attention"  : attn.squeeze().cpu().numpy(),
        })

    return results


def generate_demo_audio(demo_type):
    """Generate synthetic demo audio for testing."""
    t = np.linspace(0, CLIP_DUR, int(SR * CLIP_DUR))

    if demo_type == "safe":
        freq, noise_level = 80,  0.02
    elif demo_type == "medium":
        freq, noise_level = 300, 0.15
    else:  # hazard
        freq, noise_level = 750, 0.45

    audio = (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * freq * 2 * t) +
        noise_level * np.random.randn(len(t))
    )
    audio = (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)
    return audio, SR


# ============================================================
# LOAD MODEL — cached so it only loads once
# ============================================================
@st.cache_resource
def load_cnn_model():
    """
    Load the trained CNN model from the same folder as app.py.
    __file__ gives the path to this app.py file.
    os.path.dirname gives the folder containing app.py.
    """
    app_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, "cnn_attention_model.pt")

    if not os.path.exists(model_path):
        return None, f"Model not found at: {model_path}"

    try:
        device = torch.device("cpu")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model  = SoundSafeCNN().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔥 SoundSafe Kitchen")
        st.caption("Predicting cooking hazards before they happen")
        st.divider()

        page = st.radio(
            "Navigate to:",
            [
                "🔍 Audio Analysis",
                "📊 Hazard Timeline",
                "🏆 Model Comparison",
                "ℹ️ About the Project",
            ],
        )

        st.divider()
        st.markdown("**Model Performance (Test Set)**")
        perf = {
            "Accuracy" : "99.1%",
            "Recall"   : "98.0%",
            "Precision": "100%",
            "F1 Score" : "0.9901",
        }
        for label, val in perf.items():
            c1, c2 = st.columns(2)
            c1.caption(label)
            c2.markdown(f"**{val}**")

        st.divider()
        st.caption("CP2 — Deep Learning Track")
        st.caption("Dataset: ESC-50 + Synthetic")

    return page


# ============================================================
# PAGE 1: AUDIO ANALYSIS
# ============================================================
def page_audio_analysis(model):
    st.title("🔍 Kitchen Audio Hazard Analysis")
    st.markdown(
        "Upload a kitchen audio file or try a demo. "
        "The model predicts cooking hazards from audio alone — "
        "**no camera, no sensor, no new hardware.**"
    )

    if model is None:
        st.error(
            "⚠️ CNN model not loaded. "
            "Make sure `cnn_attention_model.pt` is in the "
            "`streamlit_app` folder."
        )
        return

    # File upload
    uploaded = st.file_uploader(
        "Upload kitchen audio (WAV, MP3, OGG, FLAC)",
        type=["wav", "mp3", "ogg", "flac"],
    )

    # Demo buttons
    st.markdown("**Or try a built-in demo:**")
    col1, col2, col3 = st.columns(3)
    demo = None
    with col1:
        if st.button("🟢 Safe Cooking Demo",
                     use_container_width=True):
            demo = "safe"
    with col2:
        if st.button("🟡 Active Boiling Demo",
                     use_container_width=True):
            demo = "medium"
    with col3:
        if st.button("🔴 Hazard Sound Demo",
                     use_container_width=True):
            demo = "hazard"

    # Resolve audio source
    audio_data, sr_audio = None, None

    if uploaded is not None:
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            audio_data, sr_audio = librosa.load(
                tmp_path, sr=None, mono=True)
            st.success(
                f"✓ Loaded: **{uploaded.name}**  |  "
                f"Duration: {len(audio_data)/sr_audio:.1f}s  |  "
                f"SR: {sr_audio} Hz"
            )
        except Exception as e:
            st.error(f"Could not load audio: {e}")
        finally:
            os.unlink(tmp_path)

    elif demo is not None:
        audio_data, sr_audio = generate_demo_audio(demo)
        names = {
            "safe"  : "Safe Cooking",
            "medium": "Active Boiling",
            "hazard": "Hazard Sound",
        }
        st.info(f"Demo audio: **{names[demo]}**")

    # Run analysis
    if audio_data is not None:
        with st.spinner("Analysing audio..."):
            results = analyse_audio(audio_data, sr_audio, model)

        if not results:
            st.warning("Audio too short. Please use at least 5 seconds.")
            return

        # Save results for Hazard Timeline page
        st.session_state["results"]    = results
        st.session_state["audio_data"] = audio_data
        st.session_state["sr"]         = sr_audio

        # Summary statistics
        max_prob = max(r["probability"] for r in results)
        avg_prob = float(np.mean([r["probability"] for r in results]))
        n_hazard = sum(
            1 for r in results if r["risk_level"] == "HAZARD")

        # Overall verdict banner
        if max_prob >= THRESHOLD:
            verdict = "⚠️ HAZARD DETECTED"
            col     = COLOR_HAZARD
            emoji   = "🔴"
        elif max_prob >= 0.30:
            verdict = "⚡ CAUTION — Monitor Closely"
            col     = COLOR_CAUTION
            emoji   = "🟡"
        else:
            verdict = "✅ SAFE — No Hazard Detected"
            col     = COLOR_SAFE
            emoji   = "🟢"

        st.markdown(
            f"<div style='background:{col}22;"
            f"border-left:6px solid {col};"
            f"padding:18px; border-radius:8px; margin:16px 0;'>"
            f"<h2 style='color:{col}; margin:0;'>{emoji} {verdict}</h2>"
            f"<p style='margin:6px 0 0 0; font-size:16px;'>"
            f"Maximum hazard probability: "
            f"<strong>{max_prob:.1%}</strong></p></div>",
            unsafe_allow_html=True,
        )

        # Four metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Probability",  f"{max_prob:.1%}")
        m2.metric("Avg Probability",  f"{avg_prob:.1%}")
        m3.metric("Hazard Windows",
                  f"{n_hazard} / {len(results)}")
        m4.metric("Duration",
                  f"{len(audio_data)/sr_audio:.1f}s")

        # Waveform plot
        st.markdown("### Audio Waveform")
        t_ax  = np.linspace(
            0, len(audio_data) / sr_audio, len(audio_data))
        step  = max(1, len(t_ax) // 2000)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=t_ax[::step], y=audio_data[::step],
            mode="lines",
            line=dict(color="#4A90D9", width=0.8),
            name="Waveform",
        ))
        # Highlight hazard windows in red
        for r in results:
            if r["risk_level"] == "HAZARD":
                fig_w.add_vrect(
                    x0=r["time_sec"],
                    x1=min(r["time_sec"] + CLIP_DUR,
                           len(audio_data) / sr_audio),
                    fillcolor=COLOR_HAZARD, opacity=0.15,
                    layer="below", line_width=0,
                )
        fig_w.update_layout(
            height=200,
            margin=dict(t=10, b=30, l=40, r=10),
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # Hazard probability over time
        st.markdown("### Hazard Probability Over Time")
        times = [r["time_sec"] + CLIP_DUR / 2 for r in results]
        probs = [r["probability"] for r in results]
        pt_cols = [
            COLOR_HAZARD  if p >= THRESHOLD else
            COLOR_CAUTION if p >= 0.30 else COLOR_SAFE
            for p in probs
        ]
        fig_p = go.Figure()
        # Background colour zones
        fig_p.add_hrect(y0=0,    y1=0.30,
                         fillcolor=COLOR_SAFE,
                         opacity=0.07, layer="below", line_width=0)
        fig_p.add_hrect(y0=0.30, y1=0.50,
                         fillcolor=COLOR_CAUTION,
                         opacity=0.07, layer="below", line_width=0)
        fig_p.add_hrect(y0=0.50, y1=1.0,
                         fillcolor=COLOR_HAZARD,
                         opacity=0.07, layer="below", line_width=0)
        fig_p.add_trace(go.Scatter(
            x=times, y=probs,
            mode="lines+markers",
            line=dict(color="white", width=2),
            marker=dict(color=pt_cols, size=10),
            name="Hazard Probability",
        ))
        fig_p.add_hline(
            y=THRESHOLD, line_dash="dash",
            line_color=COLOR_HAZARD, line_width=1.5,
            annotation_text="Alert threshold (0.5)",
        )
        fig_p.add_hline(
            y=0.30, line_dash="dot",
            line_color=COLOR_CAUTION, line_width=1,
            annotation_text="Caution (0.3)",
        )
        fig_p.update_layout(
            height=280,
            margin=dict(t=10, b=40, l=50, r=10),
            xaxis_title="Time (seconds)",
            yaxis_title="Hazard Probability",
            yaxis=dict(range=[0, 1.05]),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Attention weights
        st.markdown("### Model Attention — Where the Model Focused")
        st.caption(
            "Higher attention weight = the model considered that "
            "time frame more important for hazard prediction."
        )
        max_idx   = int(np.argmax(probs))
        best_attn = results[max_idx]["attention"]
        attn_t    = np.linspace(0, CLIP_DUR, len(best_attn))
        attn_col  = (COLOR_HAZARD
                     if probs[max_idx] >= THRESHOLD
                     else COLOR_SAFE)
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=attn_t, y=best_attn,
            fill="tozeroy",
            fillcolor=(
                "rgba(201,75,75,0.25)"   if attn_col == COLOR_HAZARD else
                "rgba(45,106,79,0.25)"
           ),  
           line=dict(color=attn_col, width=2),
           name="Attention",
    
        ))
        fig_a.update_layout(
            height=200,
            margin=dict(t=10, b=30, l=50, r=10),
            xaxis_title="Time within highest-risk window (s)",
            yaxis_title="Attention Weight",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig_a, use_container_width=True)

        # Recommendation
        st.markdown("### 💡 Recommendation")
        if max_prob >= THRESHOLD:
            st.error(
                "⚠️ **HAZARD DETECTED** — Return to the kitchen "
                "immediately. Oil may be overheating, a pan may be "
                "about to boil over, or food may be burning."
            )
        elif max_prob >= 0.30:
            st.warning(
                "⚡ **Monitor Closely** — Cooking sounds are "
                "intensifying. Keep an eye on the hob."
            )
        else:
            st.success(
                "✅ **Sounds Safe** — No hazard detected. "
                "Consistent with normal safe cooking."
            )


# ============================================================
# PAGE 2: HAZARD TIMELINE
# ============================================================
def page_hazard_timeline():
    st.title("📊 Hazard Timeline")

    if "results" not in st.session_state:
        st.info(
            "Please upload and analyse audio on the "
            "**Audio Analysis** page first."
        )
        return

    results    = st.session_state["results"]
    audio_data = st.session_state["audio_data"]
    sr         = st.session_state["sr"]

    st.markdown(
        f"Timeline for last analysed clip — "
        f"**{len(audio_data)/sr:.1f}s** duration, "
        f"**{len(results)}** windows analysed"
    )

    # Summary counts
    counts = {"Safe": 0, "Caution": 0, "HAZARD": 0}
    for r in results:
        counts[r["risk_level"]] += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Safe Windows",    counts["Safe"])
    c2.metric("🟡 Caution Windows", counts["Caution"])
    c3.metric("🔴 Hazard Windows",  counts["HAZARD"])

    # Pie chart
    fig_pie = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker_colors=[COLOR_SAFE, COLOR_CAUTION, COLOR_HAZARD],
        hole=0.4,
    ))
    fig_pie.update_layout(
        height=280,
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Window-by-window table
    st.markdown("### Window-by-Window Results")
    rows = []
    for r in results:
        emoji = (
            "🔴" if r["risk_level"] == "HAZARD" else
            "🟡" if r["risk_level"] == "Caution" else "🟢"
        )
        rows.append({
            "Time Window"    : (f"{r['time_sec']:.1f}s – "
                                f"{r['time_sec']+CLIP_DUR:.1f}s"),
            "Risk Level"     : f"{emoji} {r['risk_level']}",
            "Probability"    : f"{r['probability']:.4f}",
            "Alert Triggered": (
                "⚠️ YES" if r["probability"] >= THRESHOLD else "No"
            ),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================
def page_model_comparison():
    st.title("🏆 Model Comparison")
    st.markdown(
        "All four models evaluated on the **same 117-clip "
        "held-out test set**. Results are directly comparable."
    )

    data = {
        "Model"         : [
            "Rule-Based", "Random Forest",
            "LSTM", "1D CNN+Attention"
        ],
        "Accuracy"      : [0.9316, 0.9658, 0.9658, 0.9915],
        "Precision"     : [0.9388, 0.9796, 1.0000, 1.0000],
        "Recall"        : [0.9020, 0.9412, 0.9216, 0.9804],
        "F1 Score"      : [0.9200, 0.9600, 0.9592, 0.9901],
        "ROC-AUC"       : [0.9058, 0.9970, 0.9994, 0.9884],
        "Missed Hazards": [5, 3, 4, 1],
        "False Alarms"  : [3, 1, 0, 0],
    }
    colours = [COLOR_CAUTION, COLOR_SAFE, "#4A90D9", COLOR_HAZARD]

    # Grouped bar chart
    st.markdown("### Performance Metrics")
    fig_b = go.Figure()
    metrics_show = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for i, mdl in enumerate(data["Model"]):
        vals = [data[m][i] for m in metrics_show]
        fig_b.add_trace(go.Bar(
            name=mdl, x=metrics_show, y=vals,
            marker_color=colours[i],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))
    fig_b.update_layout(
        barmode="group", height=400,
        yaxis=dict(range=[0.87, 1.06]),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # Safety metrics
    st.markdown("### Safety Metrics")
    col1, col2 = st.columns(2)
    with col1:
        fig_fn = go.Figure(go.Bar(
            x=data["Model"],
            y=data["Missed Hazards"],
            marker_color=colours,
            text=data["Missed Hazards"],
            textposition="outside",
        ))
        fig_fn.update_layout(
            title="Missed Hazards (lower = safer)",
            height=280, yaxis=dict(range=[0, 7]),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_fn, use_container_width=True)

    with col2:
        fig_fp = go.Figure(go.Bar(
            x=data["Model"],
            y=data["False Alarms"],
            marker_color=colours,
            text=data["False Alarms"],
            textposition="outside",
        ))
        fig_fp.update_layout(
            title="False Alarms (lower = fewer unnecessary alerts)",
            height=280, yaxis=dict(range=[0, 5]),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_fp, use_container_width=True)

    # Full table
    st.markdown("### Full Results Table")
    df = pd.DataFrame(data).set_index("Model")
    st.dataframe(df, use_container_width=True)


# ============================================================
# PAGE 4: ABOUT
# ============================================================
def page_about():
    st.title("ℹ️ About SoundSafe Kitchen")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
## The Problem
Every day in the UK, **170 kitchen fires** start from unattended cooking.
Cooking is the **single largest cause** of house fires, costing
**£1 billion per year** and 49 deaths annually in England.

**The critical insight:** there is a 2–4 minute acoustic warning window
before a hazard becomes dangerous.

## The Solution
SoundSafe Kitchen uses a **1D CNN with Temporal Self-Attention**
to predict hazards from passive audio.
**No camera. No food probe. No new hardware.**

## Architecture""")
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Input | Mel Spectrogram (64 bands, 5s clips) |
| Architecture | 1D CNN × 3 blocks + Temporal Attention |
| Output | Hazard probability (0–1) |
| Threshold | 0.5 (alert triggered above this) |
| Training data | ESC-50 + Synthetic kitchen audio |
""")

    with col2:
        st.markdown("### Performance")
        st.metric("Accuracy",  "99.1%")
        st.metric("Precision", "100%")
        st.metric("Recall",    "98.0%")
        st.metric("F1 Score",  "0.9901")
        st.metric("Missed Hazards", "1 / 51")

    st.divider()
    st.markdown("""
### Why Audio Only?
- Cameras raise **privacy concerns** in home kitchens
- Food temperature probes require **physical contact**
- Smoke detectors only trigger **after** the hazard
- A microphone is already in every smart speaker, phone, and laptop

### Dataset
- **ESC-50**: 2,000 environmental sound clips across 50 categories
- **Synthetic**: Programmatically generated safe/hazard kitchen sounds
- **Total**: 585 labelled clips → 117 held-out test clips
""")


# ============================================================
# MAIN — route to the correct page
# ============================================================
model, load_err = load_cnn_model()

page = render_sidebar()

if page == "🔍 Audio Analysis":
    page_audio_analysis(model)
elif page == "📊 Hazard Timeline":
    page_hazard_timeline()
elif page == "🏆 Model Comparison":
    page_model_comparison()
elif page == "ℹ️ About the Project":
    page_about()
