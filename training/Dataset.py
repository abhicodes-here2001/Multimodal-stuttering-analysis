import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Preprocessing.audio_check import load_audio, ensure_wav_16k_mono


class StutterDataset(Dataset):
    def __init__(self, csv_path, audio_dir, label_columns):
        """
        Args:
            csv_path: Path to the CSV file with labels.
            audio_dir: Root directory of audio clips.
            label_columns: List of column names to use as labels.
        """
        self.audio_dir = audio_dir
        self.label_columns = label_columns
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"CSV has {len(df)} rows")
        
        # Filter: keep only rows where audio file EXISTS
        valid_rows = []
        missing_count = 0
        
        for idx, row in df.iterrows():
            file_path = self._build_file_path(row)
            if os.path.exists(file_path):
                valid_rows.append(idx)
            else:
                missing_count += 1
        
        # Keep only valid rows
        self.df = df.loc[valid_rows].reset_index(drop=True)
        
        print(f"Found {len(self.df)} audio files")
        print(f"Missing {missing_count} audio files (skipped)")
        
    def _build_file_path(self, row):
        """Build file path from CSV row."""
        show = row['Show'].strip()
        ep_id = int(row['EpId'])
        clip_id = int(row['ClipId'])
        return f"{self.audio_dir}/{show}/{ep_id}/{show}_{ep_id}_{clip_id}.wav"
    
    def __len__(self):
        return len(self.df)
    
    def _get_file_path(self, row):
        """Build file path from CSV row."""
        return self._build_file_path(row)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self._get_file_path(row)
        
        # Check and convert if needed (skips if already 16kHz mono)
        file_path = ensure_wav_16k_mono(file_path)
        
        # Load audio as 1D tensor
        waveform, sr = load_audio(file_path)
        
        # Get binary labels (0 -> 0, >=1 -> 1)
        labels = []
        for col in self.label_columns:
            value = row[col]
            labels.append(1 if value >= 1 else 0)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return waveform, labels