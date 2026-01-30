"""
WavLM Inference Module
======================
Uses our trained WavLM model to detect stutters in audio.

This module:
- Loads our fine-tuned WavLM model from checkpoints
- Processes audio files using existing chunking pipeline
- Returns stutter predictions with timestamps for each chunk

Stutter Types Detected (5 labels):
1. Prolongation - Stretched sounds ("Ssssssnake")
2. Block - Silent pauses/stuck moments  
3. SoundRep - Sound repetitions ("B-b-b-ball")
4. WordRep - Word repetitions ("I-I-I want")
5. Interjection - Filler words ("um", "uh", "like")
"""

import torch
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.WaveLm_model import WaveLmStutterClassification
from Preprocessing.chunking import chunk_audio_file
from Preprocessing.audio_check import load_audio
from typing import Dict, List, Optional

# Stutter type labels (same order as training)
STUTTER_LABELS = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

# Paths - use absolute path based on project root
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints/wavlm_stutter_classification_best.pth")


def load_wavlm_model(checkpoint_path: str = CHECKPOINT_PATH, device: Optional[str] = None):
    """
    Load our trained WavLM stutter classification model.
    
    Args:
        checkpoint_path: Path to the saved model weights (.pth file)
        device: Device to load model on (None = auto-detect)
    
    Returns:
        Loaded model ready for inference, and device string
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        else:
            device = "cpu"
    
    print(f"Loading WavLM model on device: {device}")
    
    # Create model with same architecture as training
    model = WaveLmStutterClassification(
        num_labels=len(STUTTER_LABELS),
        freeze_base=True,
        unfreeze_last_n_layers=1 
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout)
    
    print("WavLM model loaded successfully!")
    return model, device


def predict_chunk(waveform: torch.Tensor, model: WaveLmStutterClassification, 
                  device: str, threshold: float = 0.4) -> Dict:
    """
    Predict stutter types for a single audio chunk.
    
    Args:
        waveform: Audio tensor (1D, 16kHz)
        model: Loaded WavLM model
        device: Device model is on
        threshold: Probability threshold for positive prediction (default 0.4)
    
    Returns:
        Dict with probabilities and binary predictions for each stutter type
    """
    with torch.no_grad():
        # Prepare input: add batch dimension
        waveform = waveform.unsqueeze(0).to(device)  # (1, samples)
        
        # Get model predictions (logits)
        logits = model(waveform)
        
        # Convert to probabilities using sigmoid
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (5,)
        
        # Create result dictionary
        result = {
            'probabilities': {},
            'detected': [],
            'raw_probs': probs.tolist()
        }
        
        for i, label in enumerate(STUTTER_LABELS):
            prob = float(probs[i])
            result['probabilities'][label] = prob
            if prob >= threshold:
                result['detected'].append(label)
        
        # Explicitly label as Fluent if no stutters are detected
        if not result['detected']:
            result['label'] = "Fluent"
        else:
            result['label'] = ", ".join(result['detected'])
        
        return result


def analyze_audio(audio_path: str, model: Optional[WaveLmStutterClassification] = None,
                  device: Optional[str] = None, chunk_sec: float = 3.0,
                  hop_sec: Optional[float] = None, threshold: float = 0.4) -> Dict:
    """
    Analyze a full audio file for stutters.
    
    This is the MAIN function you'll use. It:
    1. Loads/converts audio to 16kHz mono
    2. Chunks audio into windows (using existing chunking.py)
    3. Runs WavLM model on each chunk
    4. Returns timestamped stutter predictions
    
    Args:
        audio_path: Path to audio file (any format)
        model: Pre-loaded model (optional - will load if None)
        device: Device to use
        chunk_sec: Chunk size in seconds (default 3.0)
        hop_sec: Hop between chunks (default = chunk_sec, no overlap)
        threshold: Probability threshold for detection
    
    Returns:
        Dict with:
        - 'chunks': List of chunk analyses with timestamps
        - 'summary': Overall stutter counts and stats
    """
    # Load model if not provided
    if model is None:
        model, device = load_wavlm_model()
    
    print(f"Analyzing audio: {audio_path}")
    
    # Use existing chunking pipeline
    chunks = chunk_audio_file(
        audio_path,
        chunk_sec=chunk_sec,
        hop_sec=hop_sec,
        convert_first=True  # Converts to 16kHz mono
    )
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Analyze each chunk
    results = []
    stutter_counts = {label: 0 for label in STUTTER_LABELS}
    
    for i, chunk_info in enumerate(chunks):
        waveform = chunk_info['chunk']
        start_time = chunk_info['start']
        end_time = chunk_info['end']
        
        # Get predictions for this chunk
        prediction = predict_chunk(waveform, model, device, threshold)
        
        # Update counts
        for label in prediction['detected']:
            stutter_counts[label] += 1
        
        results.append({
            'chunk_index': i,
            'start_time': start_time,
            'end_time': end_time,
            'prediction': prediction
        })
    
    # Create summary
    total_chunks = len(chunks)
    chunks_with_stutter = sum(1 for r in results if r['prediction']['detected'])
    
    summary = {
        'total_chunks': total_chunks,
        'chunks_with_stutter': chunks_with_stutter,
        'stutter_percentage': (chunks_with_stutter / total_chunks * 100) if total_chunks > 0 else 0,
        'stutter_counts': stutter_counts,
        'chunk_duration_sec': chunk_sec
    }
    
    return {
        'audio_path': audio_path,
        'chunks': results,
        'summary': summary
    }


