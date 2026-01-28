"""Chunking utilities for audio preprocessing (torchaudio-only version).

Main features:
- chunk_waveform: chunk a torch waveform into fixed-length windows with overlap
- chunk_audio_file: convert audio to 16 kHz mono (via audio_check) and chunk it, optionally saving chunk wav files
- process_and_analyze: convenience helper that converts + chunks + calls a user-provided model function on each chunk

All operations require torch and torchaudio to be installed.
"""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torchaudio

from Preprocessing.audio_check import ensure_wav_16k_mono, load_audio


def chunk_waveform(
    waveform: torch.Tensor,
    sr: int,
    chunk_sec: float = 3.0,
    hop_sec: Optional[float] = None,
) -> Iterable[Tuple[float, float, torch.Tensor]]:
    """Yield (start_sec, end_sec, chunk) for waveform.

    Args:
        waveform: 1D torch.Tensor of samples
        sr: sample rate (samples per second)
        chunk_sec: length of each chunk in seconds (default 3.0)
        hop_sec: hop between starts in seconds. If None, hop_sec == chunk_sec (no overlap).

    Yields:
        start_sec, end_sec, chunk (torch.Tensor)
    """
    if hop_sec is None:
        hop_sec = chunk_sec

    total_samples = waveform.numel()
    chunk_samples = int(round(chunk_sec * sr))
    hop_samples = int(round(hop_sec * sr))
    pos = 0
    while pos < total_samples:
        start = pos
        end = min(pos + chunk_samples, total_samples)
        chunk = waveform[start:end]
        yield start / sr, end / sr, chunk
        if end == total_samples:
            break
        pos += hop_samples


def _save_chunk_wav(chunk: torch.Tensor, sr: int, out_path: str):
    """Save chunk as a WAV file using torchaudio."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # torchaudio.save expects (channels, frames)
    tensor = chunk
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(out_path), tensor, sr, format="wav")


def chunk_audio_file(
    input_path: str,
    out_dir: Optional[str] = None,
    chunk_sec: float = 3.0,
    hop_sec: Optional[float] = None,
    convert_first: bool = True,
) -> List[dict]:
    """Convert input audio to 16 kHz mono and chunk it.

    Args:
        input_path: original audio file
        out_dir: if provided, each chunk will be saved as WAV under this directory
        chunk_sec: length of each chunk in seconds (default 3.0)
        hop_sec: hop between chunk starts (default = chunk_sec -> no overlap)
        convert_first: if True, run ensure_wav_16k_mono before chunking

    Returns:
        list of dicts: {"start": float, "end": float, "chunk": torch.Tensor, "chunk_path": Optional[str]}
    """
    inp = str(input_path)
    if convert_first:
        wav_path = ensure_wav_16k_mono(inp)
    else:
        wav_path = inp

    waveform, sr = load_audio(wav_path)

    results = []
    stem = Path(wav_path).stem

    for start, end, chunk in chunk_waveform(waveform, sr, chunk_sec=chunk_sec, hop_sec=hop_sec):
        rec = {"start": start, "end": end, "chunk": chunk, "chunk_path": None}
        if out_dir is not None:
            out_name = f"{stem}_start{start:.2f}_end{end:.2f}.wav"
            out_path = Path(out_dir) / out_name
            _save_chunk_wav(chunk, sr, out_path)
            rec["chunk_path"] = str(out_path)
        results.append(rec)

    return results


def process_and_analyze(
    input_path: str,
    model_fn: Callable[[torch.Tensor, int], object],
    chunk_sec: float = 3.0,
    hop_sec: Optional[float] = None,
    out_dir: Optional[str] = None,
    convert_first: bool = True,
) -> List[dict]:
    """Full pipeline: convert -> chunk -> run model_fn on each chunk.

    model_fn should have signature model_fn(waveform, sr) and return a serializable result (e.g., dict of predictions).

    Returns a list of entries: {start, end, model_output, chunk_path}
    """
    chunks = chunk_audio_file(
        input_path, out_dir=out_dir, chunk_sec=chunk_sec, hop_sec=hop_sec, convert_first=convert_first
    )

    outputs = []
    for rec in chunks:
        chunk = rec["chunk"]
        sr = 16000
        try:
            out = model_fn(chunk, sr)
        except Exception as e:
            out = {"error": str(e)}
        outputs.append({"start": rec["start"], "end": rec["end"], "model_output": out, "chunk_path": rec.get("chunk_path")})

    return outputs


if __name__ == "__main__":
    # Simple CLI demonstration
    import argparse

    parser = argparse.ArgumentParser(description="Chunk audio into fixed windows and optionally save chunks")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--outdir", default=None, help="Directory to save chunk WAVs")
    parser.add_argument("--chunk-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=None)
    args = parser.parse_args()

    res = chunk_audio_file(args.input, out_dir=args.outdir, chunk_sec=args.chunk_sec, hop_sec=args.hop_sec)
    for r in res:
        print(f"chunk {r['start']:.2f}-{r['end']:.2f}: saved={r['chunk_path'] is not None}")
