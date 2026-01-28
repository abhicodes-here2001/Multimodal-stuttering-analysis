"""Audio preprocessing utilities (torchaudio-only version).

Primary responsibilities:
- Accept arbitrary audio files (mp3, m4a, wav, etc.) and convert them to 16 kHz mono WAV
- Provide helpers to validate audio, load as torch tensors, and optionally cache converted files
- CLI for quick conversions

All operations require torchaudio and torch to be installed.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import torchaudio
import torch





def ensure_wav_16k_mono(input_path: str, output_path: Optional[str] = None, overwrite: bool = False) -> str:
    """Ensure the audio at input_path is a 16 kHz, mono WAV file using torchaudio only.

    Returns the path to the converted file (may be the input_path if it already meets requirements).
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = str(inp)

    outp = Path(output_path)
    if outp.parent and not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)

    if outp.exists() and not overwrite:
        return str(outp)

    # Check if already correct format
    if is_wav_16k_mono(str(inp)):
        if str(inp) != str(outp):
            shutil.copy2(str(inp), str(outp))
        return str(outp)

    # Always use torchaudio for conversion
    waveform, sr = torchaudio.load(str(inp))  # waveform: (channels, frames)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    torchaudio.save(str(outp), waveform, 16000, format="wav")
    return str(outp)


def is_wav_16k_mono(path: str) -> bool:
    """Return True if file is WAV with 16 kHz sample rate and mono channels (torchaudio only)."""
    path = str(path)
    try:
        info = torchaudio.info(path)
        sr = int(info.sample_rate)
        channels = int(info.num_channels)
        return sr == 16000 and channels == 1
    except Exception:
        return False


def load_audio(path: str):
    """Load a 16kHz mono WAV file and return (waveform, sample_rate).

    waveform is a 1-D torch.Tensor.
    Assumes the file is already 16kHz mono (use ensure_wav_16k_mono first if unsure).
    """
    path = str(path)
    waveform, sr = torchaudio.load(path)
    # Remove channel dimension to get 1D tensor
    waveform = waveform.squeeze(0)
    return waveform.float(), sr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to 16k mono WAV (torchaudio only)")
    parser.add_argument("inputs", nargs="+", help="Input audio file(s)")
    parser.add_argument("-o", "--outdir", default=None, help="Output directory (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    for inp in args.inputs:
        inp_path = Path(inp)
        if args.outdir:
            out_dir = Path(args.outdir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / (inp_path.stem + "_16k_mono.wav")
        else:
            out_file = inp_path.with_name(inp_path.stem + "_16k_mono.wav")
        try:
            out = ensure_wav_16k_mono(str(inp_path), str(out_file), overwrite=args.overwrite)
            print(f"Converted: {inp} -> {out}")
        except Exception as e:
            print(f"Failed to convert {inp}: {e}")
