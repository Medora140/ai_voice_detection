from pathlib import Path
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

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
REPO_ID = "Dora14-x/Dora_ai-voice-detector" 
FILENAME = "phase4_multicondition_mlp.pth"

print(f"Loading models on: {DEVICE}...")

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

mlp_model = DeepfakeMLP().to(DEVICE)
mlp_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
mlp_model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_NAME)
wavlm_model = WavLMModel.from_pretrained(WAVLM_NAME).to(DEVICE)
wavlm_model.eval()

print("Models loaded successfully!")

# --- 3. Audio Preprocessing ---
def preprocess_audio(audio_path, target_sr=16000, max_seconds=6):
    data, sr = sf.read(audio_path)
    waveform = torch.tensor(data, dtype=torch.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1, keepdim=True).T
    else:
        waveform = waveform.unsqueeze(0)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 0.0:
        waveform = waveform * (0.05 / rms)

    max_len = target_sr * max_seconds
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]

    return waveform.squeeze(0)

# --- 4. Prediction Function ---
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

# --- 5. UI Layout & Styling ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=EB+Garamond:wght@400;500&display=swap');
:root, .gradio-container {
  --canvas: #faf9f5; --ink: #141413; --body: #3d3d3a; --muted: #6c6a64;
  --hairline: #e6dfd8; --soft: #f5f0e8; --card: #efe9de;
  --coral: #cc785c; --coral-hover: #a9583e; --dark: #181715;
  --body-background-fill: #faf9f5; --background-fill-primary: #faf9f5;
  --background-fill-secondary: #f5f0e8; --block-background-fill: #fffefa;
  --block-label-background-fill: #fffefa; --block-label-text-color: #141413;
  --body-text-color: #3d3d3a; --block-title-text-color: #141413;
  --input-background-fill: #fffefa; --input-text-color: #141413;
  color-scheme: light !important;
}
html, body, .gradio-container, .gradio-container main { background: var(--canvas) !important; color: var(--body) !important; }
.gradio-container { width:100% !important; max-width:none !important; min-height:100vh !important; margin:0 !important; padding:0 clamp(18px, 5vw, 80px) !important; box-sizing:border-box !important; font-family:'DM Sans', Inter, sans-serif !important; }
.gradio-container main { width:100% !important; max-width:none !important; }
.gradio-container [data-testid="block-info"], .gradio-container label,
.gradio-container .wrap .label, .gradio-container .form, .gradio-container .prose,
.gradio-container .prose p, .gradio-container .prose li,
.gradio-container .panel-note, .gradio-container .privacy,
.gradio-container .footer, .gradio-container .top-label { color:var(--body) !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select,
.gradio-container [role="textbox"] { background:#fffefa !important; color:var(--ink) !important; }
.gradio-container [data-testid="audio"] { background:var(--soft) !important; color:var(--ink) !important; border-color:#d9cfc1 !important; }
.gradio-container [data-testid="audio"] * { color:inherit; }
.gradio-container [data-testid="label"] { background:#fffefa !important; color:var(--ink) !important; }
.gradio-container [data-testid="label"] *, .gradio-container .confidence, .gradio-container .label-wrap,
.gradio-container .label-wrap *, .gradio-container .wrap, .gradio-container .wrap * { color:var(--ink) !important; }
.gradio-container .progress-bar, .gradio-container progress { color:var(--coral) !important; }
.topbar { display:flex; align-items:center; justify-content:space-between; padding: 18px 0; border-bottom:1px solid var(--hairline); }
.brand { display:flex; gap:11px; align-items:center; color:var(--ink); font-weight:700; letter-spacing:-.03em; font-size:17px; }
.brand-mark { display:grid; place-items:center; width:32px; height:32px; border-radius:50%; background:var(--dark); color:var(--canvas); font-size:15px; }
.top-label { color:var(--muted); font-size:13px; }
.hero { padding:50px 0 26px; max-width:720px; }
.eyebrow { font-size:11px; letter-spacing:.15em; text-transform:uppercase; color:var(--coral-hover); font-weight:700; margin-bottom:14px; }
.hero h1 { font-family:'EB Garamond', Georgia, serif; color:var(--ink); font-size:clamp(42px,6vw,64px); font-weight:400; letter-spacing:-.035em; line-height:1.02; margin:0 0 16px; }
.hero p { font-size:16px; line-height:1.65; color:var(--muted); max-width:570px; margin:0; }
.workspace { gap:22px !important; align-items:stretch !important; }
.panel { background:#fffefa !important; color:var(--body) !important; border:1px solid var(--hairline) !important; border-radius:14px !important; padding:24px !important; box-shadow:0 8px 24px rgba(45,35,24,.035); }
.panel-title { font-size:16px; font-weight:600; color:var(--ink); margin:0 0 5px; }
.panel-note { font-size:13px; color:var(--muted); margin:0 0 18px; }
.dropzone, .dropzone > div { background:var(--soft) !important; border-color:#d9cfc1 !important; border-radius:10px !important; }
.primary-button { background:var(--coral) !important; color:white !important; border:0 !important; min-height:46px !important; border-radius:7px !important; font-weight:600 !important; }
.primary-button:hover { background:var(--coral-hover) !important; }
.result-panel { min-height:250px; }
.result-panel label, .result-panel .label, .result-panel .prose, .result-panel .prose * { color:var(--ink) !important; }
.gradio-container button.secondary { border-radius:7px !important; }
.privacy { display:flex; gap:9px; align-items:flex-start; color:var(--muted); font-size:12px; line-height:1.5; margin-top:14px; }
.privacy span { color:#5db872; font-size:14px; }
.sample-heading { font-size:12px; color:var(--muted); margin:18px 0 9px; }
.footer { display:flex; justify-content:space-between; gap:20px; margin-top:32px; padding:19px 0 28px; border-top:1px solid var(--hairline); color:var(--muted); font-size:12px; }
@media(max-width:760px) {
  .gradio-container { padding:0 16px !important; }
  .hero { padding-top:36px; }
  .panel { padding:18px !important; }
  .footer { flex-direction:column; gap:6px; }
}
"""

examples = [[str(path)] for path in sorted(Path("samples").glob("*.wav"))]

def reset_results(audio_file):
    if audio_file:
        prompt = "Audio ready. Click **Analyze voice** to see the assessment for this recording."
    else:
        prompt = "Add a recording to see whether the voice sounds authentic or synthetic."
    return None, prompt, gr.update(interactive=bool(audio_file))

with gr.Blocks(title="Voiceprint — AI voice detector", css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
      <header class="topbar">
        <div class="brand"><span class="brand-mark">V</span> voiceprint</div>
        <div class="top-label">Audio authenticity checker</div>
      </header>
      <section class="hero">
        <div class="eyebrow">A closer listen</div>
        <h1>Is this voice<br>human or synthetic?</h1>
        <p>Upload a short audio clip and let the detector look for the subtle traces left by AI-generated speech.</p>
      </section>
    """)
    with gr.Row(elem_classes="workspace"):
        with gr.Column(scale=11, elem_classes="panel"):
            gr.HTML('<div class="panel-title">Add an audio sample</div><p class="panel-note">Upload a file or record a few seconds of speech.</p>')
            audio_input = gr.Audio(
                type="filepath", label="Audio file", sources=["upload", "microphone"],
                format="wav", elem_classes="dropzone",
            )
            analyze_btn = gr.Button("Analyze voice", variant="primary", elem_classes="primary-button", interactive=False)
            gr.HTML('<div class="privacy"><span>●</span><div>Your recording is used only to produce this result. For best results, use a clear clip up to 6 seconds.</div></div>')
            if examples:
                gr.HTML('<div class="sample-heading">Or try a sample recording</div>')
                gr.Examples(examples=examples, inputs=audio_input, label="Sample clips")
        with gr.Column(scale=9, elem_classes=["panel", "result-panel"]):
            gr.HTML('<div class="panel-title">Detection result</div><p class="panel-note">Your assessment will appear here.</p>')
            label_output = gr.Label(num_top_classes=2, label="Voice classification")
            verdict_output = gr.Markdown("Add a recording to see whether the voice sounds authentic or synthetic.")

    audio_input.upload(
        fn=reset_results,
        inputs=audio_input,
        outputs=[label_output, verdict_output, analyze_btn],
    )
    audio_input.stop_recording(
        fn=reset_results,
        inputs=audio_input,
        outputs=[label_output, verdict_output, analyze_btn],
    )
    audio_input.clear(
        fn=reset_results,
        inputs=audio_input,
        outputs=[label_output, verdict_output, analyze_btn],
    )
    analyze_btn.click(fn=analyze_voice, inputs=audio_input, outputs=[label_output, verdict_output])
    gr.HTML('<footer class="footer"><span>Voiceprint · AI voice analysis</span><span>Results are model estimates and may not be conclusive.</span></footer>')

if __name__ == "__main__":
    demo.launch()