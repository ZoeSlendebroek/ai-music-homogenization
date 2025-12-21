#!/usr/bin/env python3
"""
End-to-end pipeline to:

1. Load AI Afrobeats metadata WITH features (already extracted).
2. Load human Afrobeats metadata (with preview_url).
3. Download human previews and extract the same audio features as AI.
4. Standardize features across AI + human tracks.
5. For each AI track, find nearest human tracks in feature space.

Usage example:

  python afrobeat_ai_human_match_pipeline.py \
    --ai_meta_with_features data/ai/afrobeat/meta_with_features.csv \
    --human_meta data/human/afrobeat/human_meta.csv \
    --audio_cache_dir data/human/afrobeat/audio_cache \
    --out_human_meta_with_features data/human/afrobeat/human_meta_with_features.csv \
    --out_matches data/afrobeat/ai_human_matches.csv \
    --neighbors 3 \
    --tempo_tolerance 8.0
"""

import argparse
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import librosa
import requests


# Feature extraction (shared between AI and human pipelines)
def extract_features_from_audio(file_path, sr_target=22050, max_duration_sec=30.0):
    """
    Load a local audio file and extract features with librosa.

    Returns a dict with:
      - tempo
      - spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate
      - mfcc_1..mfcc_13
      - chroma_1..chroma_12
    or None if loading fails.
    """
    try:
        # Ensure a string path is passed to librosa
        from pathlib import Path
        file_path = Path(file_path)
        y, sr = librosa.load(str(file_path), sr=sr_target, mono=True)

        target_len = int(30 * sr)

        if len(y) > target_len:
            # Take a centered 30-second window for consistency
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        else:
            y = librosa.util.fix_length(y, size=target_len)

        tempo_arr = librosa.beat.tempo(y=y, sr=sr)
        tempo = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

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


# Human preview download and feature extraction
def download_preview(url: str, dest_path: Path, timeout: int = 30):
    """
    Download preview audio from a URL into a local cache.
    """
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"    Downloading: {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"    Download failed for {url}: {e}")
        return None

    return dest_path


def extract_features_for_human_tracks(
    human_meta_path: Path,
    audio_cache_dir: Path,
    out_human_meta_with_features: Path,
):
    """
    Download human preview audio, extract features, and write an updated metadata CSV.
    """
    if not human_meta_path.exists():
        raise FileNotFoundError(f"Human meta file not found: {human_meta_path}")

    df = pd.read_csv(human_meta_path)
    if "preview_url" not in df.columns:
        raise ValueError("Human meta CSV must contain a 'preview_url' column.")

    print(f"Loaded human metadata: {len(df)} rows from {human_meta_path}")

    rows = []
    for idx, row in df.iterrows():
        track_id = row.get("track_id") or row.get("id") or f"row_{idx}"
        url = row["preview_url"]

        print(f"[{idx+1:03d}] {track_id}:", end=" ")

        if pd.isna(url) or not isinstance(url, str) or not url.strip():
            print("missing preview_url")
            continue

        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_cache_dir / f"{track_id}.m4a"

        audio_path = download_preview(url, audio_path)
        if audio_path is None or not audio_path.exists():
            print("download failed")
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
    print(f"\nExtracted features for {len(out_df)} human tracks.")
    out_human_meta_with_features.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_human_meta_with_features, index=False)
    print(f"Saved human meta with features to: {out_human_meta_with_features}")

    return out_df


# Feature handling and matching logic
def infer_feature_columns(df: pd.DataFrame):
    """
    Identify audio feature columns based on known names and prefixes.
    """
    feature_cols = []
    for col in df.columns:
        if col in {
            "tempo",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
        }:
            feature_cols.append(col)
        elif col.startswith("mfcc_") or col.startswith("chroma_"):
            feature_cols.append(col)
    return sorted(feature_cols)


def standardize_features(ai_df, human_df, feature_cols):
    """
    Z-score features using statistics computed over combined AI and human data.
    """
    combined = pd.concat(
        [ai_df[feature_cols], human_df[feature_cols]],
        ignore_index=True,
    )

    means = combined.mean(axis=0)
    stds = combined.std(axis=0).replace(0, 1.0)

    ai_scaled = (ai_df[feature_cols] - means) / stds
    human_scaled = (human_df[feature_cols] - means) / stds

    return ai_scaled.values, human_scaled.values


# Main 
def main():
    """
    Run the full AI-to-human Afrobeats matching pipeline.
    """
    ap = argparse.ArgumentParser(description="AI vs human Afrobeats matching pipeline")
    ap.add_argument("--ai_meta_with_features", required=True)
    ap.add_argument("--human_meta", required=True)
    ap.add_argument("--audio_cache_dir", required=True)
    ap.add_argument("--out_human_meta_with_features", required=True)
    ap.add_argument("--out_matches", required=True)
    ap.add_argument("--neighbors", type=int, default=3)
    ap.add_argument("--tempo_tolerance", type=float, default=8.0)
    args = ap.parse_args()

    ai_df = pd.read_csv(args.ai_meta_with_features)
    print(f"Loaded AI meta with features: {len(ai_df)} rows")

    human_feat_df = extract_features_for_human_tracks(
        Path(args.human_meta),
        Path(args.audio_cache_dir),
        Path(args.out_human_meta_with_features),
    )

    feature_cols = infer_feature_columns(ai_df)

    match_df = match_ai_to_human_smart(
        ai_df,
        human_feat_df,
        feature_cols,
        n_neighbors=args.neighbors,
        tempo_tolerance=args.tempo_tolerance,
    )

    Path(args.out_matches).parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(args.out_matches, index=False)
    print(f"Saved AI-to-human matches to: {args.out_matches}")


if __name__ == "__main__":
    main()
