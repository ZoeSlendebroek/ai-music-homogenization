#!/usr/bin/env python3
"""
Extract audio features for AI-generated Afrobeats tracks.

This script reads an AI metadata CSV (which must include a `filename` column),
loads each corresponding audio file from a provided directory, extracts a fixed
set of audio features using librosa, and writes a new CSV with those feature
columns appended.

Feature extraction is intentionally aligned with the human pipeline so that the
resulting CSV can be used for direct comparison and downstream modeling.

For each audio file, the script:
  - loads mono audio at a target sample rate
  - normalizes duration to a fixed 30-second window:
      * if the audio is longer than 30 seconds, it takes the centered 30 seconds
      * if the audio is shorter, it pads/truncates to exactly 30 seconds
  - computes tempo, MFCC means (13), chroma means (12), and common spectral stats

Usage:
  python extract_ai_features.py \
    --ai_meta ../data/ai/afrobeat/meta.csv \
    --audio_dir ../data/ai/afrobeat/audio \
    --out_meta ../data/ai/afrobeat/meta_with_features.csv
"""

import argparse
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import librosa


def extract_features_from_audio(file_path, sr_target=22050, max_duration_sec=30.0):
    """
    Load a local audio file and extract features with librosa.

    Parameters
    ----------
    file_path : str or Path
        Path to the audio file on disk.
    sr_target : int
        Target sample rate for loading audio. Audio is resampled to this rate.
    max_duration_sec : float
        Duration to standardize audio to, in seconds. Audio is either centered and
        cropped to this duration, or padded to this duration.

    Returns
    -------
    dict or None
        A dictionary containing:
          - tempo
          - spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate
          - mfcc_1..mfcc_13 (means over time)
          - chroma_1..chroma_12 (means over time)
        Returns None if loading or feature extraction fails.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)

        target_len = int(30 * sr)  # 30 seconds in samples

        if len(y) > target_len:
            # Use a consistent 30-second window by taking the centered segment.
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]

            # Alternative windowing strategy (kept as a commented option):
            # start = int(5 * sr)  # skip first 5 seconds
            # y = y[start : start + target_len]
        else:
            y = librosa.util.fix_length(y, size=target_len)

        # Tempo
        tempo_arr = librosa.beat.tempo(y=y, sr=sr)
        tempo = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # (13,)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)  # (12,)

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()

        feat = {
            "tempo": tempo,
            "spectral_centroid": float(spec_centroid),
            "spectral_bandwidth": float(spec_bandwidth),
            "spectral_rolloff": float(spec_rolloff),
            "zero_crossing_rate": float(zcr),
        }

        for i, v in enumerate(mfcc_mean, start=1):
            feat[f"mfcc_{i}"] = float(v)

        for i, v in enumerate(chroma_mean, start=1):
            feat[f"chroma_{i}"] = float(v)

        return feat

    except Exception as e:
        print(f"    Error processing {file_path}: {e}")
        return None


def main() -> None:
    """
    Parse CLI arguments, extract features for each AI track, and write an updated metadata CSV.

    The input metadata CSV must include a `filename` column. Each filename is resolved relative
    to `--audio_dir`. Files that are missing or fail feature extraction are skipped.

    The output CSV contains the original metadata columns plus the extracted feature columns.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ai_meta",
        required=True,
        help="Path to AI metadata CSV (must have at least a 'filename' column).",
    )
    ap.add_argument(
        "--audio_dir",
        required=True,
        help="Directory containing AI audio files (e.g., ai_afrobeat_XX_YY.wav).",
    )
    ap.add_argument(
        "--out_meta",
        required=True,
        help="Path to output CSV with features added.",
    )
    args = ap.parse_args()

    ai_meta_path = Path(args.ai_meta)
    audio_dir = Path(args.audio_dir)
    out_meta_path = Path(args.out_meta)

    if not ai_meta_path.exists():
        raise FileNotFoundError(f"AI meta file not found: {ai_meta_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    df = pd.read_csv(ai_meta_path)
    if "filename" not in df.columns:
        raise ValueError("AI meta CSV must contain a 'filename' column.")

    print(f"Loaded AI metadata: {len(df)} rows from {ai_meta_path}")

    rows = []
    for idx, row in df.iterrows():
        fname = row["filename"]
        audio_path = audio_dir / fname

        print(f"  [{idx + 1:03d}] {fname}:", end=" ")

        if not audio_path.exists():
            print("file not found")
            continue

        feat = extract_features_from_audio(audio_path)
        if feat is None:
            print("feature extraction failed")
            continue

        print("ok")
        meta = row.to_dict()
        meta.update(feat)
        rows.append(meta)

    out_df = pd.DataFrame(rows)
    print(f"\nExtracted features for {len(out_df)} AI tracks.")

    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_meta_path, index=False)
    print(f"Saved AI meta with features to: {out_meta_path}")


if __name__ == "__main__":
    main()
