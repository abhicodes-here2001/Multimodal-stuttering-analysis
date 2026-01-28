"""
Whisper Inference Module
========================
Transcribes audio files using OpenAI's Whisper model.

Purpose in our system:
- Convert speech to text (transcription)
- Get word-level timestamps (when each word was spoken)
- This helps us map stutter detections to actual words

Whisper Model Sizes:
- tiny: 39M params, fastest but least accurate
- base: 74M params, good balance for testing
- small: 244M params, better accuracy
- medium: 769M params, even better
- large: 1550M params, best accuracy
"""
#importing open AI whisper library
import whisper
import torch
from typing import Dict, Optional, List

#load model function
def load_whisper_model(model_size: str = "base", device: Optional[str] = None) -> whisper.Whisper:
    """
    Load the Whisper model of specified size.
    
    Args:
        model_size: Size of model ("tiny", "base", "small", "medium", "large")
    
    Returns:
        Loaded Whisper model ready for transcription

    """
    # Whisper doesn't fully support MPS (Apple GPU) due to sparse tensor operations
    # Using CPU for Whisper - it's still fast enough for inference
    device = "cpu"
    
    print(f"Loading Whisper model '{model_size}' on device: {device}")
    model = whisper.load_model(model_size, device=device)
    print("Whisper model loaded successfully!")
    
    return model

#transcription function
def transcribe_audio(audio_path: str, model: Optional[whisper.Whisper] = None, 
                     model_size: str = "base", word_timestamps: bool = True) -> Dict:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file (.wav, .mp3, etc.)
        model: Pre-loaded Whisper model (optional - will load if not provided)
        model_size: Size of model to load if model is None
        word_timestamps: If True, get timestamps for each word (crucial for us!)
    
    Returns:
        Dictionary containing:
        - 'text': Full transcription string
        - 'segments': List of segments with start/end times
        - 'language': Detected language
    """
    # Load model if not provided (allows reusing same model for multiple files)
    if model is None:
        model = load_whisper_model(model_size)
    
    # Transcribe with word-level timestamps
    # word_timestamps=True is KEY for mapping stutters to words!
    result = model.transcribe(audio_path, word_timestamps=word_timestamps)
    
    return result

#function to get words with timestamps
def get_words_with_timestamps(transcription_result: Dict) -> List[Dict]:
    """
    Extract all words with their timestamps from transcription result.
    
    This is crucial for our system - we need to know WHEN each word was spoken
    so we can match it with stutter detections from WavLM.
    
    Args:
        transcription_result: Output from transcribe_audio()
    
    Returns:
        List of dicts: [{'word': 'hello', 'start': 0.0, 'end': 0.5}, ...]
    """
    words_list = []
    
    # Whisper organizes output into segments, each segment has words
    for segment in transcription_result.get('segments', []):
        # Each segment may have word-level timestamps
        if 'words' in segment:
            for word_info in segment['words']:
                words_list.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
    
    return words_list


    
   