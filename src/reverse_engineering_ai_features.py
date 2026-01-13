#!/usr/bin/env python3
"""
reverse_engineering_ai_feature_extraction.py

Extract the SAME 67 MIR features (same logic as human previews) from AI tracks
generated via Suno and stored as audio files, then save to CSV.

- Uses the SAME ComprehensiveFeatureExtractor:
  - librosa.load(sr=22050)
  - middle 30 seconds crop OR center-pad to 30 seconds
  - same feature set and parameters

Inputs:
  1) AI audio dir (default): data/afrobeat/audio/*.(mp3|wav|m4a|flac|aac|ogg)
  2) Prompts CSV (optional but recommended): data/afrobeat/suno_music_glossary_prompts.csv
     Contains: suno_song_name, human_song_file, prompt, + audit cols

Outputs:
  1) data/afrobeat/ai_tracks_features_67.csv
  2) data/afrobeat/ai_tracks_features_67_with_prompts.csv  (left-joined on suno_song_name)

Notes:
- To ensure the join works, the AI filenames should match suno_song_name values,
  e.g. "SUNO_RE_xxx.mp3" in the audio folder.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import librosa


# ----------------------------
# Feature extraction (67 MIR) -- identical to your human script
# ----------------------------

class ComprehensiveFeatureExtractor:
    """Extract multi-dimensional features from audio files (67 numeric MIR features)."""

    def __init__(self, sr: int = 22050, n_mfcc: int = 13, n_chroma: int = 12):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def extract_all_features(self, audio_path: str) -> dict:
        """Extract feature set from audio file (exactly 30 seconds, middle crop or pad)."""
        y, sr = librosa.load(audio_path, sr=self.sr)

        # Use the middle 30 seconds for comparability with previews
        target_len = int(30 * sr)

        if len(y) > target_len:
            mid = len(y) // 2
            half = target_len // 2
            start = max(0, mid - half)
            end = start + target_len
            if end > len(y):
                end = len(y)
                start = end - target_len
            y = y[start:end]
        elif len(y) < target_len:
            pad_total = target_len - len(y)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            y = np.pad(y, (pad_left, pad_right), mode="constant")

        features = {}
        features.update(self._extract_spectral_features(y, sr))
        features.update(self._extract_timbral_features(y, sr))
        features.update(self._extract_rhythmic_features(y, sr))
        features.update(self._extract_harmonic_features(y, sr))
        features.update(self._extract_structural_features(y, sr))
        features.update(self._extract_dynamic_features(y, sr))
        features.update(self._extract_mfcc_statistics(y, sr))
        return features

    def _extract_spectral_features(self, y, sr) -> dict:
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        return {
            "spectral_centroid_mean": float(np.mean(spec_cent)),
            "spectral_centroid_std": float(np.std(spec_cent)),
            "spectral_centroid_var": float(np.var(spec_cent)),
            "spectral_bandwidth_mean": float(np.mean(spec_bw)),
            "spectral_bandwidth_std": float(np.std(spec_bw)),
            "spectral_rolloff_mean": float(np.mean(spec_rolloff)),
            "spectral_rolloff_std": float(np.std(spec_rolloff)),
            "spectral_flatness_mean": float(np.mean(spec_flatness)),
            "spectral_flatness_std": float(np.std(spec_flatness)),
            "spectral_contrast_mean": float(np.mean(spec_contrast)),
            "spectral_contrast_std": float(np.std(spec_contrast)),
        }

    def _extract_timbral_features(self, y, sr) -> dict:
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        total_energy = harmonic_energy + percussive_energy

        return {
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std": float(np.std(zcr)),
            "harmonic_ratio": harmonic_energy / total_energy if total_energy > 0 else 0.0,
            "percussive_ratio": percussive_energy / total_energy if total_energy > 0 else 0.0,
        }

    def _extract_rhythmic_features(self, y, sr) -> dict:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        if len(onset_frames) > 1:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            ioi = np.diff(onset_times)
            onset_density = len(onset_frames) / (len(y) / sr)
        else:
            ioi = np.array([0.0])
            onset_density = 0.0

        ioi_mean = float(np.mean(ioi)) if len(ioi) > 0 else 0.0
        ioi_std = float(np.std(ioi)) if len(ioi) > 0 else 0.0
        ioi_cv = (ioi_std / ioi_mean) if ioi_mean > 0 else 0.0

        return {
            "tempo": tempo,
            "onset_density": float(onset_density),
            "onset_strength_mean": float(np.mean(onset_env)),
            "onset_strength_std": float(np.std(onset_env)),
            "ioi_mean": ioi_mean,
            "ioi_std": ioi_std,
            "ioi_cv": float(ioi_cv),
        }

    def _extract_harmonic_features(self, y, sr) -> dict:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        return {
            "chroma_stft_mean": float(np.mean(chroma)),
            "chroma_stft_std": float(np.std(chroma)),
            "chroma_stft_var": float(np.var(chroma)),
            "chroma_cqt_mean": float(np.mean(chroma_cqt)),
            "chroma_cqt_std": float(np.std(chroma_cqt)),
            "tonnetz_mean": float(np.mean(tonnetz)),
            "tonnetz_std": float(np.std(tonnetz)),
        }

    def _extract_structural_features(self, y, sr) -> dict:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        _, beats = librosa.beat.beat_track(y=y, sr=sr)

        if beats is None or len(beats) < 2:
            mfcc_sync = mfcc
        else:
            mfcc_sync = librosa.util.sync(mfcc, beats)

        R = librosa.segment.recurrence_matrix(mfcc_sync, mode="affinity")

        diag_mean = float(np.mean(np.diag(R))) if R.shape[0] > 0 else 0.0
        repetition_score = float(np.mean(R) - diag_mean)

        return {
            "repetition_score": repetition_score,
            "self_similarity_mean": float(np.mean(R)) if R.size else 0.0,
            "self_similarity_std": float(np.std(R)) if R.size else 0.0,
        }

    def _extract_dynamic_features(self, y, sr) -> dict:
        rms = librosa.feature.rms(y=y)[0]
        db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        mean_sq = float(np.mean(y ** 2))
        crest = float(np.max(np.abs(y)) / np.sqrt(mean_sq)) if mean_sq > 0 else 0.0

        return {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_var": float(np.var(rms)),
            "dynamic_range_db": float(np.max(db) - np.min(db)) if db.size else 0.0,
            "crest_factor": crest,
        }

    def _extract_mfcc_statistics(self, y, sr) -> dict:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features = {}
        for i in range(self.n_mfcc):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

        features["mfcc_delta_mean"] = float(np.mean(mfcc_delta))
        features["mfcc_delta_std"] = float(np.std(mfcc_delta))
        features["mfcc_delta2_mean"] = float(np.mean(mfcc_delta2))
        features["mfcc_delta2_std"] = float(np.std(mfcc_delta2))
        return features


# ----------------------------
# AI processing
# ----------------------------

def process_ai_tracks(ai_audio_dir: Path, out_features_csv: Path) -> pd.DataFrame:
    extractor = ComprehensiveFeatureExtractor()
    ai_audio_dir = Path(ai_audio_dir)

    exts = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg"]
    audio_files = []
    for e in exts:
        audio_files.extend(ai_audio_dir.glob(e))
    audio_files = sorted(audio_files)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {ai_audio_dir}")

    rows = []
    for i, audio_file in enumerate(audio_files):
        print(f"Processing AI {i+1}/{len(audio_files)}: {audio_file.name}")
        try:
            feats = extractor.extract_all_features(str(audio_file))
            feats["filename"] = audio_file.name
            feats["label"] = "AI"
            feats["track_index"] = i
            rows.append(feats)
        except Exception as e:
            print(f"  ERROR processing {audio_file.name}: {e}")
            continue

    df = pd.DataFrame(rows)

    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_features_csv, index=False)

    print(f"\nSaved {len(df)} tracks to {out_features_csv}")
    if len(df) > 0:
        print(f"Extracted {len(df.columns) - 3} features per track (excluding filename/label/track_index)")
    return df


def merge_with_prompts(ai_df: pd.DataFrame, prompts_csv: Path, out_csv: Path) -> pd.DataFrame:
    prompts_csv = Path(prompts_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not prompts_csv.exists():
        print(f"Prompts CSV not found at {prompts_csv}; skipping merge.")
        ai_df.to_csv(out_csv, index=False)
        return ai_df

    prompts = pd.read_csv(prompts_csv)

    # Join key: AI filename should equal suno_song_name (possibly with extension differences)
    # We'll create a helper join key without extension on both sides.
    ai_df = ai_df.copy()
    prompts = prompts.copy()

    ai_df["suno_join_key"] = ai_df["filename"].astype(str).apply(lambda x: Path(x).stem)
    prompts["suno_join_key"] = prompts["suno_song_name"].astype(str).apply(lambda x: Path(x).stem)

    merged = ai_df.merge(prompts, on="suno_join_key", how="left", suffixes=("", "_promptmeta"))

    # Keep suno_song_name if present; otherwise fall back to filename stem
    if "suno_song_name" not in merged.columns:
        merged["suno_song_name"] = merged["suno_join_key"]

    merged.to_csv(out_csv, index=False)

    matched = merged["prompt"].notna().sum() if "prompt" in merged.columns else 0
    print(f"\nSaved merged AI+prompts to {out_csv}")
    print(f"Prompt matches: {matched} / {len(merged)}")
    return merged


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Default AI audio dir (adjust if your AI mp3s are elsewhere)
    AI_AUDIO_DIR = project_root / "data" / "ai" / "afrobeat" / "audio"


    # Prompts created from the human previews
    PROMPTS_CSV = project_root / "data" / "afrobeat" / "suno_music_glossary_prompts.csv"

    # Outputs
    OUT_AI_FEATURES = project_root / "data" / "afrobeat" / "ai_tracks_features_67.csv"
    OUT_AI_MERGED = project_root / "data" / "afrobeat" / "ai_tracks_features_67_with_prompts.csv"

    print("=" * 70)
    print("REVERSE ENGINEERING EXPERIMENT: AI TRACKS → 67 MIR FEATURES")
    print(f"Input dir:  {AI_AUDIO_DIR}")
    print(f"Prompts:    {PROMPTS_CSV}")
    print(f"AI feats:   {OUT_AI_FEATURES}")
    print(f"Merged:     {OUT_AI_MERGED}")
    print("=" * 70)

    ai_df = process_ai_tracks(AI_AUDIO_DIR, OUT_AI_FEATURES)
    if len(ai_df) == 0:
        print("No AI tracks processed successfully. Exiting.")
        return

    _ = merge_with_prompts(ai_df, PROMPTS_CSV, OUT_AI_MERGED)

    print("\nDONE.")
    print(f"AI tracks processed: {len(ai_df)}")


if __name__ == "__main__":
    main()

