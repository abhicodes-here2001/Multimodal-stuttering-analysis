import gradio as gr
import torch
import numpy as np
import os
import traceback
from datetime import datetime
from transformers import WavLMModel
import torch.nn as nn

print(f"APP STARTUP: {datetime.now()}")

# =============================================================================
# WHY SIGMOID INSTEAD OF SOFTMAX? - A DETAILED EXPLANATION
# =============================================================================
"""
MULTI-LABEL vs MULTI-CLASS CLASSIFICATION
==========================================

Our stutter detection is a MULTI-LABEL problem:
- A single 3-second audio chunk can have MULTIPLE stutters simultaneously
- Example: Someone might have a "Block" AND a "SoundRep" in the same chunk
- Each of the 5 stutter types is INDEPENDENT of the others

SOFTMAX (❌ NOT suitable for us):
---------------------------------
- Used for MULTI-CLASS problems where classes are MUTUALLY EXCLUSIVE
- Example: "Is this image a Cat OR a Dog?" (can't be both)
- Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
- All probabilities MUST sum to 1.0
- Problem: If we used softmax and got [0.7, 0.1, 0.1, 0.05, 0.05]:
  - It would say "70% Prolongation" but FORCE other classes to be low
  - We couldn't detect multiple stutters in one chunk!

SIGMOID (✅ CORRECT for us):
----------------------------
- Used for MULTI-LABEL problems where classes are INDEPENDENT
- Each class gets its own independent probability (0 to 1)
- Formula: sigmoid(x) = 1 / (1 + exp(-x))
- Probabilities DON'T need to sum to 1
- Example output: [0.8, 0.7, 0.2, 0.1, 0.05]
  - 80% chance of Prolongation
  - 70% chance of Block  
  - Both can be detected simultaneously!

THE TRAINING & INFERENCE FLOW:
==============================

TRAINING:
---------
1. Model outputs: LOGITS (raw scores from -∞ to +∞)
   Example: [2.5, -3.0, 0.1, -1.5, -2.0]
   
2. Loss Function: BCEWithLogitsLoss
   - "WithLogits" means it applies Sigmoid INTERNALLY
   - More numerically stable than separate Sigmoid + BCELoss
   - Compares each prediction to each ground truth label independently

INFERENCE (this file):
----------------------
1. Model outputs: LOGITS (same as training)
   Example: [2.5, -3.0, 0.1, -1.5, -2.0]
   
2. We manually apply Sigmoid to convert to probabilities:
   probs = torch.sigmoid(logits)
   Result: [0.92, 0.05, 0.52, 0.18, 0.12]
   
3. Apply threshold (e.g., 0.5) to each probability:
   - 0.92 > 0.5 → Prolongation DETECTED
   - 0.05 < 0.5 → Block NOT detected
   - 0.52 > 0.5 → SoundRep DETECTED
   - etc.

4. If NO stutters detected (all below threshold):
   → Label the chunk as "Fluent"

THRESHOLD EXPLAINED:
====================
- Default: 0.5 (theoretically neutral, since sigmoid(0) = 0.5)
- Lower threshold (0.3-0.4): More SENSITIVE, catches more stutters, but more false positives
- Higher threshold (0.6-0.7): More STRICT, fewer false positives, but might miss subtle stutters
- The slider in the UI lets users adjust this based on their needs
- SAME threshold is applied to ALL 5 classes (simplest approach)
"""

class WaveLmStutterClassification(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.hidden_size = self.wavlm.config.hidden_size
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

STUTTER_LABELS = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

STUTTER_DEFINITIONS = {
    'Prolongation': 'Sound stretched longer than normal',
    'Block': 'Complete stoppage of airflow/sound',
    'SoundRep': 'Sound/syllable repetition',
    'WordRep': 'Whole word repetition',
    'Interjection': 'Filler words like um, uh'
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

wavlm_model = None
whisper_model = None
models_loaded = False

def load_models():
    global wavlm_model, whisper_model, models_loaded
    if models_loaded:
        return True
    try:
        print("Loading WavLM...")
        wavlm_model = WaveLmStutterClassification(num_labels=5)
        checkpoint_path = "wavlm_stutter_classification_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                wavlm_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                wavlm_model.load_state_dict(checkpoint)
            print("Checkpoint loaded!")
        wavlm_model.to(device)
        wavlm_model.eval()
        
        print("Loading Whisper...")
        import whisper
        whisper_model = whisper.load_model("base", device=device)
        
        models_loaded = True
        print("Models loaded!")
        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        traceback.print_exc()
        return False

def load_audio(audio_path):
    print(f"Loading: {audio_path}")
    try:
        import librosa
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        return torch.from_numpy(waveform).float(), 16000
    except Exception as e:
        print(f"librosa error: {e}")
    try:
        import soundfile as sf
        waveform, sr = sf.read(audio_path, dtype='float32')
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(waveform).float()
        if sr != 16000:
            import torchaudio
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform.unsqueeze(0)).squeeze(0)
        return waveform, 16000
    except Exception as e:
        print(f"soundfile error: {e}")
    raise Exception("Could not load audio")

def analyze_chunk(chunk_tensor, threshold=0.5):
    with torch.no_grad():
        logits = wavlm_model(chunk_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    detected = [STUTTER_LABELS[i] for i, p in enumerate(probs) if p > threshold]
    return detected, dict(zip(STUTTER_LABELS, probs.tolist()))

def analyze_audio(audio_input, threshold, progress=gr.Progress()):
    print(f"\n=== ANALYZE CLICKED ===")
    print(f"Input: {audio_input}, Type: {type(audio_input)}, Threshold: {threshold}")
    
    progress(0, desc="🔄 Starting analysis...")
    
    if audio_input is None:
        return "⚠️ Please upload an audio file first!", "", "", ""
    
    audio_path = audio_input
    if isinstance(audio_input, tuple):
        import tempfile, soundfile as sf
        sr, data = audio_input
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(f.name, data, sr)
        audio_path = f.name
    
    if not os.path.exists(audio_path):
        return f"File not found: {audio_path}", "", "", ""
    
    print(f"File: {audio_path}, Size: {os.path.getsize(audio_path)}")
    
    try:
        progress(0.1, desc="🔄 Loading models...")
        if not models_loaded and not load_models():
            return "❌ Failed to load models", "", "", ""
        
        progress(0.2, desc="🎵 Loading audio file...")
        waveform, sr = load_audio(audio_path)
        duration = len(waveform) / sr
        print(f"Duration: {duration:.1f}s")
        
        progress(0.3, desc="✂️ Splitting audio into chunks...")
        chunk_samples = int(3.0 * sr)
        stutter_counts = {l: 0 for l in STUTTER_LABELS}
        timeline = []
        
        total_chunks = (len(waveform) + chunk_samples - 1) // chunk_samples
        
        for i, start in enumerate(range(0, len(waveform), chunk_samples)):
            progress(0.3 + (0.4 * i / total_chunks), desc=f"🔍 Analyzing chunk {i+1}/{total_chunks}...")
            
            end = min(start + chunk_samples, len(waveform))
            chunk = waveform[start:end]
            if len(chunk) < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
            
            detected, _ = analyze_chunk(chunk, threshold)
            for l in detected:
                stutter_counts[l] += 1
            timeline.append({"time": f"{start/sr:.1f}-{end/sr:.1f}s", "detected": detected or ["Fluent"]})
        
        progress(0.75, desc="🗣️ Transcribing with Whisper...")
        print("Running Whisper...")
        transcription = whisper_model.transcribe(audio_path).get('text', '')
        
        progress(0.9, desc="📊 Generating report...")
        total = sum(stutter_counts.values())
        summary = f"## ✅ Analysis Complete!\n\n**Duration:** {duration:.1f}s\n**Total Stutters Detected:** {total}\n\n### Stutter Counts:\n"
        for l, c in stutter_counts.items():
            emoji = "🔴" if c > 0 else "⚪"
            summary += f"- {emoji} **{l}**: {c}\n"
        
        timeline_md = "| Time | Detected |\n|---|---|\n"
        for t in timeline[:15]:
            timeline_md += f"| {t['time']} | {', '.join(t['detected'])} |\n"
        if len(timeline) > 15:
            timeline_md += f"\n*...and {len(timeline) - 15} more chunks*"
        
        defs = "## 📖 Stutter Type Definitions\n\n"
        defs += "\n".join([f"**{k}:** {v}" for k, v in STUTTER_DEFINITIONS.items()])
        
        progress(1.0, desc="✅ Done!")
        print("Done!")
        return summary, transcription, timeline_md, defs
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return f"Error: {e}\n\n{traceback.format_exc()}", "", "", ""

print("Building UI...")

with gr.Blocks(title="Stutter Analysis", css="""
    .loading-text { 
        font-size: 1.2em; 
        color: #666; 
        padding: 20px;
        text-align: center;
    }
""") as demo:
    gr.Markdown("""
    # 🎙️ Speech Fluency Analysis System
    
    Upload an audio file to analyze stuttering patterns using AI (WavLM + Whisper).
    
    **Supported formats:** WAV, MP3, M4A, FLAC, OGG
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="🎤 Upload Audio", type="filepath")
            threshold = gr.Slider(
                minimum=0.3, 
                maximum=0.7, 
                value=0.5, 
                step=0.05,
                label="Detection Threshold",
                info="Lower = more sensitive, Higher = more strict"
            )
            btn = gr.Button("🔍 Analyze Speech", variant="primary", size="lg")
            gr.Markdown("*Analysis takes 30-60 seconds depending on audio length*")
        
        with gr.Column(scale=2):
            summary = gr.Markdown(value="### 👆 Upload audio and click Analyze to start")
    
    with gr.Tabs():
        with gr.TabItem("📝 Transcription"):
            trans = gr.Markdown()
        with gr.TabItem("📈 Timeline"):
            timeline = gr.Markdown()
        with gr.TabItem("📖 Definitions"):
            defs = gr.Markdown()
    
    gr.Markdown("""
    ---
    **Note:** The spinner will appear while processing. Please wait for analysis to complete.
    """)
    
    # The show_progress parameter shows a spinner during processing
    btn.click(
        fn=analyze_audio, 
        inputs=[audio, threshold], 
        outputs=[summary, trans, timeline, defs],
        show_progress="full"  # Shows loading spinner
    )

print("Loading models...")
load_models()

print("Launching...")
demo.queue()
demo.launch(ssr_mode=False)
