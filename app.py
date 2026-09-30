import gradio as gr
import numpy as np
from pathlib import Path
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from huggingface_hub import hf_hub_download

# --- 1. Architecture Definition ---
class DeepfakeMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

# --- 2. Load Models (Cached at startup) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAVLM_NAME = "microsoft/wavlm-base"

# Replace 'your-username/ai-voice-detector' with your actual HF repo name
REPO_ID = "Dora14-x/Dora_ai-voice-detector" 
FILENAME = "phase4_multicondition_mlp.pth"

print(f"Loading models on: {DEVICE}...")

# This automatically downloads the checkpoint from HF and caches it locally!
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# Load MLP
mlp_model = DeepfakeMLP().to(DEVICE)
mlp_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
mlp_model.eval()

# Load WavLM
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_NAME)
wavlm_model = WavLMModel.from_pretrained(WAVLM_NAME).to(DEVICE)
wavlm_model.eval()

print("Models loaded successfully!")

# --- 3. Audio Standardization ---
def preprocess_audio(audio_path, target_sr=16000, max_seconds=6):
    # Using soundfile for robust local file loading across OS platforms
    data, sr = sf.read(audio_path)
    waveform = torch.tensor(data, dtype=torch.float32)

    # Multi-channel to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1, keepdim=True).T
    else:
        waveform = waveform.unsqueeze(0)

    # Uniform resampling to 16kHz
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # RMS Loudness Normalization
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 0.0:
        waveform = waveform * (0.05 / rms)

    # Truncate
    max_len = target_sr * max_seconds
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]

    return waveform.squeeze(0)

# --- 4. Prediction Pipeline ---
def analyze_voice(audio_file):
    if audio_file is None:
        return None, "Please upload or record an audio file to analyze."

    try:
        waveform = preprocess_audio(audio_file)
        inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = wavlm_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            logit = mlp_model(embedding)
            fake_prob = torch.sigmoid(logit).item()

        real_prob = 1.0 - fake_prob

        # Prepare UI outputs
        confidences = {
            "Authentic (Real)": real_prob,
            "Synthetic (Deepfake)": fake_prob
        }

        if fake_prob >= 0.5:
            summary = (
                f"🚨 **Verdict: SYNTHETIC SPEECH (DEEPFAKE)**\n\n"
                f"Confidence: **{fake_prob * 100:.1f}%** synthetic artifacts detected."
            )
        else:
            summary = (
                f"✅ **Verdict: AUTHENTIC SPEECH (REAL)**\n\n"
                f"Confidence: **{real_prob * 100:.1f}%** natural voice characteristics."
            )

        return confidences, summary

    except Exception as e:
        return None, f"Error analyzing audio: {str(e)}"

# --- 5. Gradio Dashboard Design ---
with gr.Blocks(title="AI Voice Anti-Spoofing Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Cross-Lingual AI Voice Deepfake Detector
        Upload a `.wav` file or record a voice snippet to test for synthetic generation artifacts (Voice Conversion & TTS).
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath", 
                label="Audio Source (Upload or Record)", 
                sources=["upload", "microphone"]
            )
            analyze_btn = gr.Button("Analyze Audio", variant="primary")

            # Clickable pre-loaded examples if you placed them in samples/
            example_files = [str(p) for p in Path("samples").glob("*.wav")]
            if example_files:
                gr.Examples(examples=example_files, inputs=audio_input)

        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label="Classification Probabilities")
            verdict_markdown = gr.Markdown("Submit an audio sample to see the assessment.")

    analyze_btn.click(
        fn=analyze_voice,
        inputs=audio_input,
        outputs=[label_output, verdict_markdown]
    )

if __name__ == "__main__":
    demo.launch()