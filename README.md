# 🎙️ Cross-Lingual AI Voice Deepfake Detector

An end-to-end audio anti-spoofing pipeline engineered to detect synthetic speech and zero-shot voice cloning attacks across multiple languages, including English, Hindi, Kannada, and Konkani.

The detector targets modern acoustic manipulation attacks, covering both Voice Conversion (FreeVC) and Autoregressive Text-to-Speech (XTTS-v2).

---

##  Highlights & Key Findings

* **Backbone Feature Extractor:** Employs a frozen `microsoft/wavlm-base` self-supervised model to capture fine-grained phase, spectral, and temporal artifacts from raw speech.
* **Non-Linear Classification:** Uses a custom PyTorch Multi-Layer Perceptron (MLP) with dropout regularization to map WavLM representations to authenticity probabilities.
* **Overcoming Generalization Failure:** Early linear probes trained strictly on Voice Conversion suffered complete collapse on unseen TTS attacks (~46.2% EER). Migrating to an MLP with **Multi-Condition Training** dropped out-of-domain error to **4.2% EER** on held-out XTTS-v2 data while preserving **1.7% EER** on FreeVC attacks.
* **Acoustic Standardization:** Built-in safeguards enforce uniform 16 kHz resampling and RMS loudness normalization (target 0.05 RMS) to prevent models from learning trivial pipeline shortcuts.

---

## 📊 Benchmark Results

| Model Architecture | Training Condition | FreeVC Evaluation (EER) | Unseen XTTS-v2 (OOD EER) |
| :--- | :--- | :--- | :--- |
| Logistic Regression Probe | FreeVC-only | ~1.1% | 46.2% (Random chance) |
| PyTorch MLP (Phase 3) | FreeVC-only | 1.0% | 45.0% (Overfit to engine) |
| **PyTorch MLP (Phase 4)** | **Multi-Condition (FreeVC + XTTS)** | **1.7%** | **4.2% (Robust)** |

*EER: Equal Error Rate (lower is better).*

---

## 📁 Repository Structure

```text
ai_voice_detect_ui/
├── models/
│   └── phase4_multicondition_mlp.pth   # Trained PyTorch MLP checkpoint (~1 MB)
├── samples/                            # Audio samples for rapid UI testing (.wav)
├── app.py                              # Interactive Gradio web application
├── test_suite.py                       # Automated batch audio evaluation script
├── requirements.txt                    # Project dependencies
└── README.md
```

## Quickstart & Setup

### Prerequisites
Python 3.10 or 3.11 installed on your machine.

Git installed and configured.

A working microphone (optional, for live recording tests).

Step-by-Step Installation Points
Clone the Repository:

Download the codebase locally to your workspace:

Bash
git clone [https://github.com/](https://github.com/)<your-username>/ai-voice-detector-ui.git
cd ai-voice-detector-ui
Create and Activate an Isolated Virtual Environment:

Keeps project dependencies isolated from system Python:

Bash
python -m venv venv
Activate the virtual environment based on your operating system:

Windows (PowerShell):

PowerShell
.\venv\Scripts\Activate.ps1
Linux / macOS:

Bash
source venv/bin/activate
Install Required Dependencies:

Installs PyTorch, Hugging Face Transformers, Gradio, and audio parsing tools:

Bash
pip install -r requirements.txt
Acquire Model Weights:

Ensure the trained weights checkpoint (phase4_multicondition_mlp.pth) is placed inside the models/ directory:

Plaintext
models/phase4_multicondition_mlp.pth

Alternative (Automated): If configured with the Hugging Face Hub snippet, app.py will automatically download and cache the checkpoint from your repository on first run.

Prepare Test Audio Samples (Optional):

Place reference .wav files into the samples/ directory so they appear as clickable examples inside the Gradio interface.

## Running the Application
1. **Launch the Gradio Web Dashboard**
Start the local server:

Bash
python app_ui.py
Open your browser at http://127.0.0.1:7860 to access the drag-and-drop audio detector and microphone interface.

2. **Run Automated Batch Benchmarks**
To evaluate an entire directory of mixed audio files (.wav, .mp3, .flac) directly from the CLI:

Bash
python test_suite.py --test-dir samples/

## Datasets Used
Authentic Voices: Vaani Dataset (IISc / ARTPARK / Google) covering English, Hindi, Kannada, and Konkani.

In-Domain Synthetics: Zero-shot voice conversion clones generated with FreeVC24.

Out-of-Domain Synthetics: Autoregressive multi-lingual speech clones generated with Coqui XTTS-v2.
