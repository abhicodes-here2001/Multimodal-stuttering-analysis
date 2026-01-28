"""
Hugging Face Spaces - Gradio App for Stutter Analysis
=====================================================
This is a standalone Gradio app for deployment on Hugging Face Spaces.

To deploy:
1. Create a new Space on huggingface.co/spaces
2. Choose "Gradio" as SDK
3. Upload this folder's contents
4. Add your model checkpoint to the Space
"""

import gradio as gr
import torch
import torchaudio
import tempfile
import os
import json
from datetime import datetime
from transformers import WavLMModel
import torch.nn as nn
import whisper

# ============================================================================
# MODEL DEFINITION (same as models/WaveLm_model.py)
# ============================================================================

class WaveLmStutterClassification(nn.Module):
    def __init__(self, num_labels=5, freeze_encoder=True, unfreeze_last_n_layers=1):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.hidden_size = self.wavlm.config.hidden_size
        
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            
            if unfreeze_last_n_layers > 0:
                for layer in self.wavlm.encoder.layers[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        self.num_labels = num_labels
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# ============================================================================
# STUTTER LABELS & DEFINITIONS
# ============================================================================

STUTTER_LABELS = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

STUTTER_DEFINITIONS = {
    'Prolongation': 'Sound stretched longer than normal (e.g., "Ssssssnake")',
    'Block': 'Complete stoppage of airflow/sound with tension',
    'SoundRep': 'Sound/syllable repetition (e.g., "B-b-b-ball")',
    'WordRep': 'Whole word repetition (e.g., "I-I-I want")',
    'Interjection': 'Filler words like "um", "uh", "like"'
}

SEVERITY_THRESHOLDS = {'very_mild': 5, 'mild': 10, 'moderate': 20, 'severe': 30}

# ============================================================================
# GLOBAL MODEL LOADING
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
wavlm_model = None
whisper_model = None

def load_models():
    global wavlm_model, whisper_model
    
    # Load WavLM
    print("Loading WavLM model...")
    wavlm_model = WaveLmStutterClassification(num_labels=5)
    
    # Try to load checkpoint
    checkpoint_path = "wavlm_stutter_classification_best.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        wavlm_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint with {checkpoint.get('val_accuracy', 'N/A')} accuracy")
    else:
        print("WARNING: No checkpoint found, using random weights")
    
    wavlm_model.to(device)
    wavlm_model.eval()
    
    # Load Whisper
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base", device=device)
    
    print("Models loaded!")

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def preprocess_audio(audio_path):
    """Convert audio to 16kHz mono"""
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0), 16000

def chunk_audio(waveform, sr, chunk_sec=3.0):
    """Split audio into chunks"""
    chunk_samples = int(chunk_sec * sr)
    chunks = []
    
    for start in range(0, len(waveform), chunk_samples):
        end = min(start + chunk_samples, len(waveform))
        chunk = waveform[start:end]
        
        # Pad if needed
        if len(chunk) < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
        
        chunks.append({
            'chunk': chunk,
            'start': start / sr,
            'end': end / sr
        })
    
    return chunks

def analyze_chunk(chunk_waveform, threshold=0.5):
    """Run WavLM on a single chunk"""
    with torch.no_grad():
        input_tensor = chunk_waveform.unsqueeze(0).to(device)
        logits = wavlm_model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    detected = [STUTTER_LABELS[i] for i, p in enumerate(probs) if p > threshold]
    probabilities = {STUTTER_LABELS[i]: float(probs[i]) for i in range(len(STUTTER_LABELS))}
    
    return {'detected': detected, 'probabilities': probabilities}

def get_severity(word_stutter_rate):
    """Calculate severity from word stutter rate"""
    if word_stutter_rate < SEVERITY_THRESHOLDS['very_mild']:
        return 'Very Mild', 1
    elif word_stutter_rate < SEVERITY_THRESHOLDS['mild']:
        return 'Mild', 2
    elif word_stutter_rate < SEVERITY_THRESHOLDS['moderate']:
        return 'Moderate', 3
    elif word_stutter_rate < SEVERITY_THRESHOLDS['severe']:
        return 'Severe', 4
    else:
        return 'Very Severe', 5

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_audio(audio_file, threshold=0.5):
    """Main analysis function for Gradio"""
    
    if wavlm_model is None:
        load_models()
    
    if audio_file is None:
        return "Please upload an audio file", "", "", ""
    
    try:
        # Preprocess
        waveform, sr = preprocess_audio(audio_file)
        duration = len(waveform) / sr
        
        # Chunk and analyze with WavLM
        chunks = chunk_audio(waveform, sr)
        
        stutter_counts = {label: 0 for label in STUTTER_LABELS}
        timeline = []
        
        for chunk_info in chunks:
            result = analyze_chunk(chunk_info['chunk'], threshold)
            for label in result['detected']:
                stutter_counts[label] += 1
            
            timeline.append({
                'time': f"{chunk_info['start']:.1f}s - {chunk_info['end']:.1f}s",
                'detected': ', '.join(result['detected']) if result['detected'] else 'Clear',
                'probs': result['probabilities']
            })
        
        # Transcribe with Whisper
        whisper_result = whisper_model.transcribe(audio_file, word_timestamps=True)
        transcription = whisper_result['text']
        
        # Get word-level info
        words = []
        if 'segments' in whisper_result:
            for seg in whisper_result['segments']:
                if 'words' in seg:
                    words.extend(seg['words'])
        
        # Map stutters to words
        words_with_stutter = 0
        annotated_words = []
        
        for word_info in words:
            word_start = word_info.get('start', 0)
            word_end = word_info.get('end', 0)
            word_text = word_info.get('word', '')
            
            word_stutters = []
            for chunk_info in chunks:
                if word_start < chunk_info['end'] and word_end > chunk_info['start']:
                    result = analyze_chunk(chunk_info['chunk'], threshold)
                    word_stutters.extend(result['detected'])
            
            word_stutters = list(set(word_stutters))
            if word_stutters:
                words_with_stutter += 1
                annotated_words.append(f"**[{word_text}]**({', '.join(word_stutters)})")
            else:
                annotated_words.append(word_text)
        
        # Calculate metrics
        total_words = len(words) if words else 1
        word_stutter_rate = (words_with_stutter / total_words) * 100
        severity_label, severity_score = get_severity(word_stutter_rate)
        
        # Format outputs
        summary = f"""
## 📊 Analysis Summary

**Duration:** {duration:.1f} seconds  
**Total Words:** {total_words}  
**Words with Stutters:** {words_with_stutter} ({word_stutter_rate:.1f}%)

### Severity: {severity_label} ({severity_score}/5)

### Stutter Type Counts:
"""
        for label, count in stutter_counts.items():
            if count > 0:
                summary += f"- **{label}**: {count} occurrences\n"
        
        # Annotated transcription
        annotated_text = " ".join(annotated_words) if annotated_words else transcription
        
        # Timeline
        timeline_text = "| Time | Detected Stutters |\n|------|-------------------|\n"
        for t in timeline[:15]:  # Limit to 15 rows
            timeline_text += f"| {t['time']} | {t['detected']} |\n"
        
        # Definitions
        definitions = "## 📖 Stutter Type Definitions\n\n"
        for label, desc in STUTTER_DEFINITIONS.items():
            definitions += f"**{label}:** {desc}\n\n"
        
        return summary, annotated_text, timeline_text, definitions
        
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="🎙️ Stutter Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Speech Fluency Analysis System
    
    Upload an audio file to analyze stuttering patterns using AI.
    
    **Supported formats:** WAV, MP3, M4A, FLAC
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            threshold_slider = gr.Slider(
                minimum=0.3,
                maximum=0.7,
                value=0.5,
                step=0.05,
                label="Detection Threshold",
                info="Lower = more sensitive, Higher = more conservative"
            )
            analyze_btn = gr.Button("🔍 Analyze Speech", variant="primary")
        
        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Summary")
    
    with gr.Tabs():
        with gr.Tab("📝 Transcription"):
            transcription_output = gr.Markdown(label="Annotated Transcription")
        
        with gr.Tab("📈 Timeline"):
            timeline_output = gr.Markdown(label="Timeline Analysis")
        
        with gr.Tab("📖 Definitions"):
            definitions_output = gr.Markdown(label="Stutter Definitions")
    
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input, threshold_slider],
        outputs=[summary_output, transcription_output, timeline_output, definitions_output]
    )
    
    gr.Markdown("""
    ---
    **Disclaimer:** This tool is for educational/research purposes. 
    Consult a qualified speech-language pathologist for clinical diagnosis.
    
    Built with WavLM + Whisper | [GitHub](https://github.com/abhicodes-here2001/Multimodal-stuttering-analysis)
    """)

# Load models on startup
load_models()

if __name__ == "__main__":
    demo.launch()
