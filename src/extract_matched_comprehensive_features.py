from pathlib import Path
import pandas as pd
from one_extract_comprehensive_features import ComprehensiveFeatureExtractor  # adjust import

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input/output locations (update these if your folder layout changes)
AI_AUDIO_DIR = PROJECT_ROOT / "data/ai/afrobeat/audio"
HUMAN_CACHE_DIR = PROJECT_ROOT / "data/human/afrobeat/audio_cache"
MATCHES_CSV = PROJECT_ROOT / "data/afrobeat/ai_human_matches.csv"
OUT_CSV = PROJECT_ROOT / "data/afrobeat/matched_pairs_comprehensive.csv"


def main() -> None:
    """
    Load AI-human match metadata, extract comprehensive audio features for each pair, and write results to CSV.

    This script reads a CSV of matched AI and human tracks, then for each row:
      - extracts features from the AI audio file referenced by `ai_filename`
      - extracts features from the human cached audio file referenced by `human_track_id`
    It appends a few identifying fields (side, pair_key, track_id, filename) to each feature dict and
    saves the combined result as a single CSV.

    All extraction logic and output columns are preserved; this only improves readability and comments.
    """
    matches = pd.read_csv(MATCHES_CSV)

    extractor = ComprehensiveFeatureExtractor()
    rows = []

    # Process each matched AI/human pair and collect two feature rows (one for each side).
    for _, row in matches.iterrows():
        # AI side: use the filename directly from the matches file.
        ai_filename = row["ai_filename"]  # e.g. "ai_afrobeat_05_02.wav"
        ai_path = AI_AUDIO_DIR / ai_filename

        feats_ai = extractor.extract_all_features(str(ai_path))
        feats_ai["side"] = "AI"
        feats_ai["pair_key"] = row["ai_filename"]  # or row["ai_index"]
        feats_ai["track_id"] = row["ai_track_id"]
        feats_ai["filename"] = ai_filename
        rows.append(feats_ai)

        # Human side: locate the cached .m4a by `human_track_id`.
        human_id = row["human_track_id"]  # e.g. "0aLub9xeQALt9du2squxRG"
        human_path = HUMAN_CACHE_DIR / f"{human_id}.m4a"

        feats_h = extractor.extract_all_features(str(human_path))
        feats_h["side"] = "Human"
        feats_h["pair_key"] = row["ai_filename"]
        feats_h["track_id"] = human_id
        feats_h["filename"] = row["human_title"]  # or human_id, up to you
        rows.append(feats_h)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved matched pair features to {OUT_CSV}")


if __name__ == "__main__":
    main()
