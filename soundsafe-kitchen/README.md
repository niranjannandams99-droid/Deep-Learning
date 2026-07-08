🔊 SoundSafe Kitchen
### Kitchen Hazard Prediction from Passive Audio using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soundsafe-kitchen.streamlit.app)

---

## 👨‍💻 Author
**Niranjan Nandam**
📧 your.email@gmail.com

---

## 🎯 What This Does

SoundSafe Kitchen uses a **1D CNN with Temporal Self-Attention** to predict
cooking hazards from passive kitchen audio — before they become dangerous.

- **No camera required** — audio only
- **No new hardware** — uses any phone/laptop/smart speaker microphone
- **Predicts 30–90 seconds before** the hazard occurs
- **99.1% accuracy** on held-out test set

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.1% |
| Precision | 100% (zero false alarms) |
| Recall | 98.0% (caught 50/51 hazards) |
| F1 Score | 0.9901 |
| Inference Speed | 3.2ms per clip |

---

## 🏗️ Architecture
Audio Input (any microphone)
↓
Preprocessing (16kHz → mono → filter → normalise)
↓
Mel-Spectrogram (64 bands × 157 time frames)
↓
1D CNN Block 1 (64 filters, kernel=5)
↓
1D CNN Block 2 (128 filters, kernel=3)
↓
1D CNN Block 3 (64 filters, kernel=3)
↓
Temporal Self-Attention
↓
Fully Connected → Sigmoid → Hazard Probability (0–1)

## 🚀 Run Locally

```bash
git clone https://github.com/NiranjanNandam/soundsafe-kitchen.git
cd soundsafe-kitchen
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

soundsafe-kitchen/
├── app.py                    # Streamlit application
├── cnn_attention_model.pt    # Trained CNN model (2.3 MB)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # App theme configuration
└── README.md

## 📊 Dataset

- **ESC-50** — Environmental Sound Classification (Creative Commons)
- **Synthetic** — Programmatically generated kitchen audio
- **Total**: 776 clips | 543 train / 116 val / 117 test

---

## 🔮 Future Roadmap

1. Real kitchen audio collection (replace synthetic data)
2. Multi-hazard classification (fire vs gas vs spill)
3. Edge deployment (TensorFlow Lite / mobile)
4. Smart home integration (Alexa / Google Home)

---

## 📄 Licence

MIT Licence — free to use, modify, and distribute with attribution.

---
