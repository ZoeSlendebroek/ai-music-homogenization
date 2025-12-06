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


# ============================================================
# 1. FEATURE EXTRACTION (reuse logic from your AI script)
# ============================================================

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
        # 🔧 ensure we always pass a string path to librosa / audioread
        from pathlib import Path
        file_path = Path(file_path)
        y, sr = librosa.load(str(file_path), sr=sr_target, mono=True)

        target_len = int(30 * sr)  # 30 seconds in samples
        
        if len(y) > target_len:
            # Use consistent middle 30s
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
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
        print(f"    ❌ Error processing {file_path}: {e}")
        return None


# ============================================================
# 2. DOWNLOAD HUMAN PREVIEWS + EXTRACT FEATURES
# ============================================================

def download_preview(url: str, dest_path: Path, timeout: int = 30):
    """
    Download preview audio from URL to dest_path if not already present.
    """
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"    ⬇️  Downloading: {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"    ❌ Download failed for {url}: {e}")
        return None

    return dest_path


def extract_features_for_human_tracks(
    human_meta_path: Path,
    audio_cache_dir: Path,
    out_human_meta_with_features: Path,
):
    """
    Given a CSV with columns at least: track_id, preview_url,
    download previews, extract features, and write a new CSV
    with feature columns added.
    """
    if not human_meta_path.exists():
        raise FileNotFoundError(f"Human meta file not found: {human_meta_path}")

    df = pd.read_csv(human_meta_path)
    if "preview_url" not in df.columns:
        raise ValueError("Human meta CSV must contain a 'preview_url' column.")

    print(f"📊 Loaded human metadata: {len(df)} rows from {human_meta_path}")

    rows = []
    for idx, row in df.iterrows():
        track_id = row.get("track_id") or row.get("id") or f"row_{idx}"
        url = row["preview_url"]

        print(f"[{idx+1:03d}] {track_id}:", end=" ")

        if pd.isna(url) or not isinstance(url, str) or not url.strip():
            print("❌ missing preview_url")
            continue

        # Name cached file by track_id (or fallback)
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
    print(f"\n✅ Extracted features for {len(out_df)} human tracks.")
    out_human_meta_with_features.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_human_meta_with_features, index=False)
    print(f"💾 Saved human meta with features to: {out_human_meta_with_features}")

    return out_df


# ============================================================
# 3. MATCH AI TRACKS TO HUMAN TRACKS
# ============================================================

def infer_feature_columns(df: pd.DataFrame):
    """
    Infer which columns are feature columns based on known prefixes.
    """
    feature_cols = []
    for col in df.columns:
        if col in [
            "tempo",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
        ]:
            feature_cols.append(col)
        elif col.startswith("mfcc_") or col.startswith("chroma_"):
            feature_cols.append(col)
    return sorted(feature_cols)


def standardize_features(ai_df: pd.DataFrame,
                         human_df: pd.DataFrame,
                         feature_cols):
    """
    Z-score features using mean/std computed over combined AI + human data.
    Returns (ai_scaled, human_scaled).
    """
    combined = pd.concat(
        [ai_df[feature_cols], human_df[feature_cols]],
        ignore_index=True
    )

    means = combined.mean(axis=0)
    stds = combined.std(axis=0).replace(0, 1.0)  # avoid division by zero

    ai_scaled = (ai_df[feature_cols] - means) / stds
    human_scaled = (human_df[feature_cols] - means) / stds

    return ai_scaled.values, human_scaled.values


def match_ai_to_human_smart(
    ai_df: pd.DataFrame,
    human_df: pd.DataFrame,
    feature_cols,
    n_neighbors: int = 1,
    tempo_tolerance: float = 8.0,
    max_uses_per_human: int = 1,
):
    """
    Globally match AI tracks to human tracks with reuse limits.

    - Each AI can have up to `n_neighbors` human matches.
    - Each human can be used at most `max_uses_per_human` times.
    - Uses greedy global assignment sorted by feature-distance.
    """

    # Drop rows with missing feature values
    ai_df = ai_df.dropna(subset=feature_cols).reset_index(drop=True)
    human_df = human_df.dropna(subset=feature_cols).reset_index(drop=True)

    # Standardize features
    ai_feat, human_feat = standardize_features(ai_df, human_df, feature_cols)

    ai_tempo = ai_df["tempo"].values
    human_tempo = human_df["tempo"].values

    n_ai = len(ai_df)
    n_h = len(human_df)

    # --- 1) Compute full tempo diff and distance matrices ---
    # tempo_diff[i, j] = |tempo_ai[i] - tempo_human[j]|
    tempo_diff = np.abs(ai_tempo[:, None] - human_tempo[None, :])

    # dists[i, j] = Euclidean distance in standardized feature space
    diffs = ai_feat[:, None, :] - human_feat[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    LARGE = 1e9

    # Apply tempo tolerance: disallow pairs outside tolerance by setting huge distance
    dists_masked = dists.copy()
    dists_masked[tempo_diff > tempo_tolerance] = LARGE

    # --- 2) Build candidate list within tolerance ---
    candidates = []
    for i in range(n_ai):
        for j in range(n_h):
            if dists_masked[i, j] < LARGE:
                candidates.append((dists_masked[i, j], i, j, tempo_diff[i, j]))

    # If some AIs would get nothing, we'll relax tempo in a second pass
    # but first do the "within tolerance" greedy assignment.

    # --- 3) Greedy global assignment within tempo tolerance ---
    candidates.sort(key=lambda x: x[0])  # sort by distance ascending

    ai_used = np.zeros(n_ai, dtype=int)
    human_used = np.zeros(n_h, dtype=int)

    # For rank per AI
    ai_rank_counter = [0] * n_ai

    match_rows = []

    print("\n🔗 Global matching (within tempo tolerance)...")
    for dist, i, j, td in candidates:
        if ai_used[i] >= n_neighbors:
            continue
        if human_used[j] >= max_uses_per_human:
            continue

        ai_rank_counter[i] += 1
        rank = ai_rank_counter[i]

        ai_row = ai_df.iloc[i]
        human_row = human_df.iloc[j]

        ai_track_id = (ai_row.get("id") or ai_row.get("track_id")
                       or ai_row.get("filename") or f"ai_{i}")
        human_track_id = (human_row.get("track_id") or human_row.get("id")
                          or human_row.get("filename") or f"human_{j}")

        row = {
            "ai_index": int(i),
            "ai_track_id": ai_track_id,
            "ai_filename": ai_row.get("filename", ""),
            "ai_tempo": float(ai_tempo[i]),

            "human_index": int(j),
            "human_track_id": human_track_id,
            "human_title": human_row.get("title", ""),
            "human_artist": human_row.get("artist", ""),
            "human_tempo": float(human_tempo[j]),
            "tempo_diff": float(td),
            "feature_distance": float(dist),
            "rank": rank,
        }
        match_rows.append(row)

        ai_used[i] += 1
        human_used[j] += 1

    # --- 4) Check if some AI tracks didn't get enough matches ---
    remaining_ai = [i for i in range(n_ai) if ai_used[i] < n_neighbors]

    if remaining_ai:
        print("\n⚠️  Some AI tracks have too few matches within tempo tolerance.")
        print("   Relaxing tempo constraint for remaining AIs...")

        # Build a new candidate list without tempo masking,
        # but only for remaining AIs and humans with capacity left.
        candidates_relaxed = []
        for i in remaining_ai:
            for j in range(n_h):
                if human_used[j] >= max_uses_per_human:
                    continue
                candidates_relaxed.append((dists[i, j], i, j, tempo_diff[i, j]))

        candidates_relaxed.sort(key=lambda x: x[0])

        for dist, i, j, td in candidates_relaxed:
            if ai_used[i] >= n_neighbors:
                continue
            if human_used[j] >= max_uses_per_human:
                continue

            ai_rank_counter[i] += 1
            rank = ai_rank_counter[i]

            ai_row = ai_df.iloc[i]
            human_row = human_df.iloc[j]

            ai_track_id = (ai_row.get("id") or ai_row.get("track_id")
                           or ai_row.get("filename") or f"ai_{i}")
            human_track_id = (human_row.get("track_id") or human_row.get("id")
                              or human_row.get("filename") or f"human_{j}")

            row = {
                "ai_index": int(i),
                "ai_track_id": ai_track_id,
                "ai_filename": ai_row.get("filename", ""),
                "ai_tempo": float(ai_tempo[i]),

                "human_index": int(j),
                "human_track_id": human_track_id,
                "human_title": human_row.get("title", ""),
                "human_artist": human_row.get("artist", ""),
                "human_tempo": float(human_tempo[j]),
                "tempo_diff": float(td),
                "feature_distance": float(dist),
                "rank": rank,
            }
            match_rows.append(row)

            ai_used[i] += 1
            human_used[j] += 1

    match_df = pd.DataFrame(match_rows)
    return match_df


# ============================================================
# 4. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="AI vs human Afrobeats matching pipeline")
    ap.add_argument(
        "--ai_meta_with_features",
        required=True,
        help="CSV with AI metadata and audio features (output of your AI feature script).",
    )
    ap.add_argument(
        "--human_meta",
        required=True,
        help="CSV with human metadata, including 'preview_url'.",
    )
    ap.add_argument(
        "--audio_cache_dir",
        required=True,
        help="Directory to cache downloaded human preview audio files.",
    )
    ap.add_argument(
        "--out_human_meta_with_features",
        required=True,
        help="Output CSV path for human metadata with features.",
    )
    ap.add_argument(
        "--out_matches",
        required=True,
        help="Output CSV path for AI->human matches.",
    )
    ap.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help="Number of nearest human neighbors per AI track.",
    )
    ap.add_argument(
        "--tempo_tolerance",
        type=float,
        default=8.0,
        help="Max BPM difference for candidate human tracks (before fallback).",
    )
    args = ap.parse_args()

    ai_meta_path = Path(args.ai_meta_with_features)
    human_meta_path = Path(args.human_meta)
    audio_cache_dir = Path(args.audio_cache_dir)
    out_human_meta_with_features = Path(args.out_human_meta_with_features)
    out_matches_path = Path(args.out_matches)

    if not ai_meta_path.exists():
        raise FileNotFoundError(f"AI meta-with-features file not found: {ai_meta_path}")
    if not human_meta_path.exists():
        raise FileNotFoundError(f"Human meta file not found: {human_meta_path}")

    # --- Step 1: load AI meta with features ---
    ai_df = pd.read_csv(ai_meta_path)
    print(f"📊 Loaded AI meta with features: {len(ai_df)} rows from {ai_meta_path}")

    # --- Step 2: extract human features (download previews etc.) ---
    human_feat_df = extract_features_for_human_tracks(
        human_meta_path,
        audio_cache_dir,
        out_human_meta_with_features,
    )

    if human_feat_df.empty:
        raise RuntimeError("No human tracks with successfully extracted features. Aborting.")

    # --- Step 3: infer feature columns ---
    feature_cols = infer_feature_columns(ai_df)
    if not feature_cols:
        raise RuntimeError("Could not infer audio feature columns from AI dataframe.")

    missing_in_human = [c for c in feature_cols if c not in human_feat_df.columns]
    if missing_in_human:
        raise RuntimeError(
            f"Human feature dataframe is missing columns: {missing_in_human}"
        )

    print(f"\n📐 Using feature columns ({len(feature_cols)}): {feature_cols}")

    # --- Step 4: match AI -> human ---
    match_df = match_ai_to_human_smart(
        ai_df,
        human_feat_df,
        feature_cols=feature_cols,
        n_neighbors=args.neighbors,
        tempo_tolerance=args.tempo_tolerance,
        max_uses_per_human=1,
    )

    out_matches_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(out_matches_path, index=False)
    print(f"\n💾 Saved AI->human matches to: {out_matches_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
