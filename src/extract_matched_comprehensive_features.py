from pathlib import Path
import pandas as pd
from one_extract_comprehensive_features import ComprehensiveFeatureExtractor  # adjust import

project_root = Path(__file__).resolve().parent.parent

AI_AUDIO_DIR      = project_root / "data/ai/afrobeat/audio"
HUMAN_CACHE_DIR   = project_root / "data/human/afrobeat/audio_cache"
MATCHES_CSV       = project_root / "data/afrobeat/ai_human_matches.csv"
OUT_CSV           = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"

def main():
    matches = pd.read_csv(MATCHES_CSV)

    extractor = ComprehensiveFeatureExtractor()
    rows = []

    # --- AI side (use ai_filename in your matches) ---
    for _, row in matches.iterrows():
        ai_filename = row["ai_filename"]   # e.g. "ai_afrobeat_05_02.wav"
        ai_path = AI_AUDIO_DIR / ai_filename

        feats_ai = extractor.extract_all_features(str(ai_path))
        feats_ai["side"] = "AI"
        feats_ai["pair_key"] = row["ai_filename"]   # or row["ai_index"]
        feats_ai["track_id"] = row["ai_track_id"]
        feats_ai["filename"] = ai_filename
        rows.append(feats_ai)

        # --- Human side (use human_track_id to find the .m4a) ---
        human_id = row["human_track_id"]           # e.g. "0aLub9xeQALt9du2squxRG"
        human_path = HUMAN_CACHE_DIR / f"{human_id}.m4a"

        feats_h = extractor.extract_all_features(str(human_path))
        feats_h["side"] = "Human"
        feats_h["pair_key"] = row["ai_filename"]
        feats_h["track_id"] = human_id
        feats_h["filename"] = row["human_title"]   # or human_id, up to you
        rows.append(feats_h)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved matched pair features to {OUT_CSV}")

if __name__ == "__main__":
    main()
