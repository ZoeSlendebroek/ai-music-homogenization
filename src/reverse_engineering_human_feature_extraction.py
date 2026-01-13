#!/usr/bin/env python3
"""
reverse_engineering_feature_extraction.py

(a) Extract the SAME 67 MIR features from all HUMAN sample previews (~60) located at:
    data/human/afrobeat/audio_cache/*.(m4a|wav|mp3|flac|aac|ogg)
    and save to CSV.

(b) Generate more specific, Suno-friendly prompts using a template and musical vocabulary
    inspired by Suno's "Music Glossary" (tempo terms like Adagio/Andante/Allegro/Presto,
    plus groove/dynamics/loop/compression/reverb hints).

Outputs:
1) data/afrobeat/human_preview_features_67.csv
2) data/afrobeat/suno_music_glossary_prompts.csv  (Suno title, human filename, prompt)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re
import numpy as np
import pandas as pd
import librosa


# ----------------------------
# Feature extraction (67 MIR)
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

        # Use the middle 30 seconds for comparability with iTunes previews
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
# Prompt generation (minimal, research-oriented)
# ----------------------------


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def tempo_term_from_bpm(bpm: float) -> str:
    """
    Light use of Suno glossary terms, only where it helps.
    Ranges (from your glossary):
      Adagio: 66–76
      Andante: 76–108
      Allegro: 120–168
      Presto: 168–200
    We handle gaps gently.
    """
    if bpm < 66:
        return "Adagio"
    if 66 <= bpm < 76:
        return "Adagio"
    if 76 <= bpm <= 108:
        return "Andante"
    if 108 < bpm < 120:
        return "Andante"
    if 120 <= bpm < 168:
        return "Allegro"
    if 168 <= bpm <= 200:
        return "Presto"
    return "Presto"


def rhythm_density_phrase(onset_density: float) -> str:
    """
    Uses the continuous value, but maps to 3 stable phrases.
    This is not a dataset-relative bucket; it's a simple global heuristic.
    (Typical onset_density for beat-driven music often lands around ~2–8,
     but your corpus may differ—these are conservative thresholds.)
    """
    if onset_density < 3.0:
        return "spacious groove"
    if onset_density < 6.5:
        return "steady groove"
    return "busy, percussion-forward groove"


def timing_phrase(ioi_cv: float) -> str:
    """
    ioi_cv = std(ioi) / mean(ioi).
    Low -> very consistent; higher -> more irregular.
    We keep language cautious: 'tight' vs 'syncopated feel'.
    """
    if ioi_cv < 0.45:
        return "tight pocket"
    if ioi_cv < 0.85:
        return "groovy pocket"
    return "syncopated feel"


def balance_phrase(percussive_ratio: float, harmonic_ratio: float) -> str:
    """
    Uses HPSS energy split to choose a single, simple descriptor.
    """
    # ensure valid
    percussive_ratio = max(0.0, min(1.0, percussive_ratio))
    harmonic_ratio = max(0.0, min(1.0, harmonic_ratio))

    diff = percussive_ratio - harmonic_ratio
    if diff > 0.20:
        return "drum-forward"
    if diff < -0.20:
        return "melodic"
    return "balanced drums and melody"


def loop_phrase(repetition_score: float, self_similarity_mean: float) -> str:
    """
    Structural cue. Keep it minimal + robust.
    """
    # repetition_score can be small; self_similarity_mean often more stable.
    if self_similarity_mean > 0.45 or repetition_score > 0.08:
        return "loop-based and hypnotic"
    if self_similarity_mean > 0.30 or repetition_score > 0.03:
        return "hooky and repetitive"
    return "with subtle variation"


def energy_phrase(rms_mean: float) -> str:
    """
    Optional: adds only ONE word-level constraint.
    Beware: absolute RMS depends on preprocessing; use conservative cutoffs.
    If you prefer, remove this completely.
    """
    if rms_mean < 0.055:
        return "chill"
    if rms_mean < 0.095:
        return "smooth"
    return "driving"


def clean_title_from_filename(filename: str, prefix: str = "SUNO_RE") -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "track"
    return f"{prefix}_{stem[:50]}"


def features_to_suno_prompt(row: pd.Series, include_energy: bool = True) -> str:
    """
    Minimal prompt template targeting the biggest controllable axes:
      - BPM
      - groove density + timing feel
      - percussive vs melodic balance
      - loopiness / repetition
      - (optional) energy word

    Designed to steer outputs toward the same MIR feature space as the human previews
    without over-constraining or introducing lots of prompt noise.
    """
    bpm = _safe_float(row.get("tempo", 100.0), 100.0)
    bpm_round = int(round(bpm)) if bpm > 0 else 100
    tempo_term = tempo_term_from_bpm(bpm_round)

    onset_density = _safe_float(row.get("onset_density", 0.0), 0.0)
    ioi_cv = _safe_float(row.get("ioi_cv", 0.6), 0.6)

    percussive_ratio = _safe_float(row.get("percussive_ratio", 0.5), 0.5)
    harmonic_ratio = _safe_float(row.get("harmonic_ratio", 0.5), 0.5)

    repetition_score = _safe_float(row.get("repetition_score", 0.0), 0.0)
    self_sim = _safe_float(row.get("self_similarity_mean", 0.0), 0.0)

    rms_mean = _safe_float(row.get("rms_mean", 0.08), 0.08)

    groove = rhythm_density_phrase(onset_density)
    timing = timing_phrase(ioi_cv)
    balance = balance_phrase(percussive_ratio, harmonic_ratio)
    loopiness = loop_phrase(repetition_score, self_sim)

    # Optional single energy adjective (keep it small)
    energy = energy_phrase(rms_mean) if include_energy else None

    # 1–2 sentences, user-like, specific, not jargon-heavy.
    if include_energy:
        prompt = (
            f"Instrumental afrobeats at {bpm_round} BPM ({tempo_term}), {energy}. "
            f"{groove} with a {timing}, {balance}, {loopiness}. "
            f"No vocals."
        )
    else:
        prompt = (
            f"Instrumental afrobeats at {bpm_round} BPM ({tempo_term}). "
            f"{groove} with a {timing}, {balance}, {loopiness}. "
            f"No vocals."
        )

    return prompt


def create_music_glossary_csv(
    df_features: pd.DataFrame,
    out_csv: Path,
    include_energy: bool = True
) -> pd.DataFrame:
    """
    Save minimal Suno prompts CSV.

    Columns:
      - suno_song_name
      - human_song_file
      - prompt

    Also includes a few audit columns so you can inspect whether prompts
    correlate with the intended MIR drivers.
    """
    if df_features is None or df_features.empty:
        raise ValueError("df_features is empty; cannot create prompts.")

    out = pd.DataFrame()
    out["human_song_file"] = df_features["filename"].astype(str)
    out["suno_song_name"] = out["human_song_file"].apply(lambda f: clean_title_from_filename(f, prefix="SUNO_RE"))
    out["prompt"] = df_features.apply(lambda r: features_to_suno_prompt(r, include_energy=include_energy), axis=1)

    # Audit columns (keep it small + relevant to prompt steering)
    audit_cols = [
        "tempo",
        "onset_density",
        "ioi_cv",
        "percussive_ratio",
        "harmonic_ratio",
        "repetition_score",
        "self_similarity_mean",
        "rms_mean",
    ]
    for c in audit_cols:
        if c in df_features.columns:
            out[c] = df_features[c]

    out.to_csv(out_csv, index=False)
    return out


def process_human_previews(audio_dir: Path, out_features_csv: Path) -> pd.DataFrame:
    extractor = ComprehensiveFeatureExtractor()
    audio_dir = Path(audio_dir)

    exts = ["*.m4a", "*.wav", "*.mp3", "*.flac", "*.aac", "*.ogg"]
    audio_files = []
    for e in exts:
        audio_files.extend(audio_dir.glob(e))
    audio_files = sorted(audio_files)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {audio_dir}")

    rows = []
    for i, audio_file in enumerate(audio_files):
        print(f"Processing Human {i+1}/{len(audio_files)}: {audio_file.name}")
        try:
            feats = extractor.extract_all_features(str(audio_file))
            feats["filename"] = audio_file.name
            feats["label"] = "Human"
            feats["track_index"] = i
            rows.append(feats)
        except Exception as e:
            print(f"  ERROR processing {audio_file.name}: {e}")
            continue

    df = pd.DataFrame(rows)
    # Ensure output dir exists
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_features_csv, index=False)
    print(f"\nSaved {len(df)} tracks to {out_features_csv}")
    if len(df) > 0:
        print(f"Extracted {len(df.columns) - 3} features per track (excluding filename/label/track_index)")
    return df


def main():
    project_root = Path(__file__).resolve().parent.parent

    # previews are here :
    HUMAN_PREVIEW_DIR = project_root / "data" / "human" / "afrobeat" / "audio_cache"

    # Outputs
    OUT_FEATURES = project_root / "data" / "afrobeat" / "human_preview_features_67.csv"
    OUT_PROMPTS = project_root / "data" / "afrobeat" / "suno_music_glossary_prompts.csv"

    print("=" * 70)
    print("REVERSE ENGINEERING EXPERIMENT: HUMAN PREVIEWS → 67 MIR → SUNO PROMPTS")
    print(f"Input dir:  {HUMAN_PREVIEW_DIR}")
    print(f"Features:   {OUT_FEATURES}")
    print(f"Prompts:    {OUT_PROMPTS}")
    print("=" * 70)

    df = process_human_previews(HUMAN_PREVIEW_DIR, OUT_FEATURES)
    if len(df) == 0:
        print("No tracks processed successfully. Exiting.")
        return

    _ = create_music_glossary_csv(df, OUT_PROMPTS)

    print("\nDONE.")
    print(f"Tracks processed: {len(df)}")
    print("Next step: Use 'suno_song_name' + 'prompt' from suno_music_glossary_prompts.csv in Suno.")


if __name__ == "__main__":
    main()
