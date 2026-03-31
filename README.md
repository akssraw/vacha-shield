# Vacha_Shield

**AI-powered deepfake voice detection for real-time call protection.**

Vacha-Shield listens to live calls, detects AI-cloned voices using a custom neural network, transcribes speech in Hindi/Hinglish/English, and alerts users to potential scams — all in real time.

---

## What It Does

- **Deepfake Detection** — Detects AI-cloned and synthetic voices using a CNN neural network with PCEN + Delta spectrograms
- **Live Call Monitoring** — Continuously analyzes microphone audio during calls
- **Live Transcription** — Transcribes Hindi, Hinglish, and English speech using Sarvam AI
- **Scam Keyword Detection** — Automatically flags dangerous phrases like OTP, KYC, UPI PIN, remote access
- **Continuous Learning** — Users submit labeled clips via the web UI to retrain the model

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For microphone tools:

```bash
pip install -r requirements-optional.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
SARVAM_API_KEY=your_sarvam_key_here
```

### 3. Run the backend

```bash
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Analysis Profiles

| Profile | Sensitivity | Best For |
|---------|-------------|----------|
| `conservative` | Low | Minimize false positives |
| `balanced` | Medium | General use |
| `strict` | High | Default |
| `forensic` | Very High | Deep investigation |

---

## Call Monitor

Runs in terminal and listens to live microphone audio:

```bash
python call_monitor.py
```

**Features:**
- Deepfake detection every 4 seconds
- Live Hindi/Hinglish/English transcription (requires `SARVAM_API_KEY`)
- Automatic scam keyword flagging
- Saves suspicious clips to `flagged_calls/`

---

## Training

### Train on ASVspoof 2019

```bash
python train_model.py \
  --dataset_dir data/ASVspoof2019_LA_train \
  --protocol_file data/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt \
  --epochs 20
```

### Continuous Learning

Collect samples via the web UI feedback buttons, then:

```bash
python train_knowledge_base.py
```

---

## Deployment

### Docker

```bash
docker build -t vacha-shield .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e SARVAM_API_KEY=your_key \
  vacha-shield
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq LLM key for semantic detection |
| `SARVAM_API_KEY` | — | Sarvam AI key for live transcription |
| `DEFAULT_ANALYSIS_PROFILE` | `strict` | Detection profile |
| `ALERT_MIN_THRESHOLD` | `0.62` | Minimum probability to trigger alert |
| `SEMANTIC_OVERRIDE_ENABLED` | `false` | Enable LLM-based AI self-identification |
| `PORT` | `5000` | Server port |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python · Flask · Flask-SocketIO · Gunicorn |
| ML | PyTorch · Librosa · NumPy · scikit-learn |
| Audio | PyAudio · SoundFile · imageio-ffmpeg |
| Transcription | Sarvam AI (saaras:v3) |
| LLM | Groq (llama-3.3-70b) |
| Frontend | React · TypeScript · Vite · Tailwind CSS |
| Deployment | Docker |

---

## Built By

**Aksshat S. Rawat** — India Innovates 2026
