#!/usr/bin/env python3
"""
OVERALL DIVERSITY: HUMAN vs AI AFROBEATS (OpenL3 embeddings)

This version uses self-supervised OpenL3 embeddings instead of hand-crafted
librosa features. For each track:

    - Load audio
    - Take the CENTER 30 seconds (trim or zero-pad as needed)
    - Extract OpenL3 music embeddings (mel256, 512-D)
    - Average across time to get a single 512-D vector per track

Then:

    - Standardize embeddings jointly (human + AI)
    - Compute diversity metrics for each group:
        * Mean pairwise distance
        * Dispersion (distance to centroid)
        * Log-volume of covariance
        * Entropy of PCA spectrum
    - Compare the distributions of pairwise distances:
        * Mann–Whitney U test
        * Cohen's d
    - Save a PCA visualization and a JSON with all metrics.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import openl3
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# OpenL3 EMBEDDING EXTRACTION
def extract_openl3_embedding(
    filepath,
    sr_target=48000,
    duration_sec=30.0,
    input_repr="mel256",
    content_type="music",
    embedding_size=512,
):
    """
    Extract a single OpenL3 embedding vector for a track.

    Steps:
      1) Load audio mono @ sr_target
      2) Take CENTER duration_sec segment (trim or pad)
      3) Run openl3.get_audio_embedding
      4) Average over time dimension to get a 1 x D vector

    Returns:
      np.ndarray of shape (embedding_size, ) or None on failure.
    """
    try:
        y, sr = librosa.load(filepath, sr=sr_target, mono=True)

        # Center 30 seconds window
        target_len = int(duration_sec * sr)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        else:
            y = librosa.util.fix_length(y, size=target_len)

        emb, ts = openl3.get_audio_embedding(
            y,
            sr,
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
            center=False,   # we already centered the segment ourselves
            hop_size=0.5,
        )
        # emb shape: (T, D). Average over time frames.
        emb_mean = emb.mean(axis=0)
        return emb_mean.astype(np.float32)

    except Exception as e:
        print(f"     OpenL3 failed for {filepath}: {e}")
        return None


# DIVERSITY METRICS
def compute_diversity_metrics(X_norm, label):
    """
    Compute multiple diversity metrics for a collection in a normalized
    embedding space X_norm (n_samples, n_features).
    """
    n = len(X_norm)

    # 1. Mean pairwise distance
    distances = pdist(X_norm, metric="euclidean")
    mean_dist = float(distances.mean())
    std_dist = float(distances.std())

    # 2. Dispersion: average distance to centroid
    centroid = X_norm.mean(axis=0)
    centroid_dists = np.linalg.norm(X_norm - centroid, axis=1)
    dispersion = float(centroid_dists.mean())

    # 3. "Volume": log determinant of covariance matrix
    #    (approximation of hyper-ellipsoid volume)
    cov_matrix = np.cov(X_norm.T)
    sign, logdet = np.linalg.slogdet(cov_matrix)
    volume = float(logdet if sign > 0 else -np.inf)

    # 4. Entropy in PCA space (how evenly variance is distributed)
    pca = PCA()
    pca.fit(X_norm)
    var_ratios = pca.explained_variance_ratio_
    entropy = float(-np.sum(var_ratios * np.log(var_ratios + 1e-10)))

    print(f"\n{label}:")
    print(f"  Mean pairwise distance: {mean_dist:.3f} ± {std_dist:.3f}")
    print(f"  Dispersion (dist to centroid): {dispersion:.3f}")
    print(f"  Log-volume (covariance det): {volume:.3f}")
    print(f"  Entropy (PCA spectrum): {entropy:.3f}")
    print(f"  Number of tracks: {n}")

    return {
        "mean_pairwise_dist": mean_dist,
        "std_pairwise_dist": std_dist,
        "dispersion": dispersion,
        "volume": volume,
        "entropy": entropy,
        "n_tracks": n,
        "all_distances": distances,  # numpy array
    }


def plot_cov_ellipse(ax, data, color, alpha=0.3):
    """
    Plot an ellipse representing the covariance in 2D PCA space.
    """
    mean = data.mean(axis=0)
    cov = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.sqrt(eigvals)

    # Width/height are 2 * sqrt(eigvals) * k; here k=2 for ~95% if Gaussian
    width = 4 * eigvals[0]
    height = 4 * eigvals[1]
    angle = np.rad2deg(np.arccos(eigvecs[0, 0]))

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=2,
        alpha=alpha,
    )
    ax.add_patch(ell)


# MAIN ANALYSIS
def main():
    print("=" * 72)
    print("OVERALL DIVERSITY: HUMAN vs AI AFROBEATS (OpenL3 embeddings)")
    print("=" * 72)
    print("Research Question: Which collection has more internal variety in a")
    print("                   high-level musical embedding space?\n")

    # Paths (adapt to your project if needed)
    human_meta_path = Path("../data/human/afrobeat/meta.csv")
    human_audio_dir = Path("../data/human/afrobeat/audio")
    ai_meta_path = Path("../data/ai/afrobeat/meta.csv")
    ai_audio_dir = Path("../data/ai/afrobeat/audio")
    out_dir = Path("../results/similarity_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Load metadata -----------------
    human_meta = pd.read_csv(human_meta_path)
    ai_meta = pd.read_csv(ai_meta_path)

    if "filename" not in human_meta.columns:
        raise ValueError("Human meta CSV must contain a 'filename' column.")
    if "filename" not in ai_meta.columns:
        raise ValueError("AI meta CSV must contain a 'filename' column.")

    print(f"📁 Human meta: {len(human_meta)} rows from {human_meta_path}")
    print(f"📁 AI meta   : {len(ai_meta)} rows from {ai_meta_path}\n")

    # ----------------- Extract embeddings -----------------
    print("🎧 Extracting OpenL3 embeddings for HUMAN tracks...")
    human_embeddings = []
    human_filenames = []

    for idx, row in human_meta.iterrows():
        fname = row["filename"]
        audio_path = human_audio_dir / fname
        print(f"  [H {idx+1:02d}/{len(human_meta):02d}] {fname}...", end=" ")

        if not audio_path.exists():
            print("file not found, skipping")
            continue

        emb = extract_openl3_embedding(audio_path)
        if emb is None:
            print("failed")
            continue

        human_embeddings.append(emb)
        human_filenames.append(fname)
        print("✓")

    print(f"\nGot embeddings for {len(human_embeddings)} human tracks.")

    print("\nExtracting OpenL3 embeddings for AI tracks...")
    ai_embeddings = []
    ai_filenames = []

    for idx, row in ai_meta.iterrows():
        fname = row["filename"]
        audio_path = ai_audio_dir / fname
        print(f"  [A {idx+1:02d}/{len(ai_meta):02d}] {fname}...", end=" ")

        if not audio_path.exists():
            print("file not found, skipping")
            continue

        emb = extract_openl3_embedding(audio_path)
        if emb is None:
            print("failed")
            continue

        ai_embeddings.append(emb)
        ai_filenames.append(fname)
        print("✓")

    print(f"\nGot embeddings for {len(ai_embeddings)} AI tracks.\n")

    if len(human_embeddings) < 2 or len(ai_embeddings) < 2:
        raise ValueError("Need at least 2 tracks in each group to compute diversity.")

    # Convert to arrays
    human_X_raw = np.vstack(human_embeddings)
    ai_X_raw = np.vstack(ai_embeddings)

    print(f"   Human embedding shape: {human_X_raw.shape}")
    print(f"   AI embedding shape   : {ai_X_raw.shape}")

    # ----------------- Joint standardization -----------------
    print("\n📏 Standardizing embeddings jointly (Human + AI)...")
    all_X = np.vstack([human_X_raw, ai_X_raw])
    scaler = StandardScaler()
    all_X_norm = scaler.fit_transform(all_X)

    n_human = len(human_X_raw)
    human_X = all_X_norm[:n_human]
    ai_X = all_X_norm[n_human:]

    print(f"   Normalized human shape: {human_X.shape}")
    print(f"   Normalized AI shape   : {ai_X.shape}")

    # ----------------- Diversity metrics -----------------
    print("\n" + "=" * 72)
    print("DIVERSITY METRICS (OpenL3 space)")
    print("=" * 72)

    human_metrics = compute_diversity_metrics(human_X, "HUMAN TRACKS")
    ai_metrics = compute_diversity_metrics(ai_X, "AI TRACKS")

    # ----------------- Statistical comparison -----------------
    print("\n" + "=" * 72)
    print("STATISTICAL COMPARISON (pairwise distances)")
    print("=" * 72)

    h_dists = human_metrics["all_distances"]
    a_dists = ai_metrics["all_distances"]

    stat, p_val = mannwhitneyu(h_dists, a_dists, alternative="two-sided")

    human_mean = float(h_dists.mean())
    ai_mean = float(a_dists.mean())
    diff = human_mean - ai_mean

    # Cohen's d (pooled SD)
    pooled_sd = np.sqrt((h_dists.std() ** 2 + a_dists.std() ** 2) / 2)
    effect_size = float(diff / pooled_sd)

    print(f"\nMann–Whitney U test (all pairwise distances):")
    print(f"  Human mean: {human_mean:.3f}")
    print(f"  AI mean   : {ai_mean:.3f}")
    print(f"  Difference (Human - AI): {diff:.3f}")
    print(f"  U-statistic: {stat:.1f}")
    print(f"  p-value   : {p_val:.4e}")
    print(f"  Cohen's d : {effect_size:.3f}")

    diversity_ratio = human_mean / ai_mean
    print(f"\nDiversity ratio (Human / AI mean distance): {diversity_ratio:.3f}")

    # Interpretation
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    if p_val < 0.05:
        if human_mean > ai_mean:
            print("✓ Human tracks show SIGNIFICANTLY MORE diversity than AI tracks")
            print("  → In OpenL3 embedding space, AI Afrobeats are more HOMOGENEOUS.")
        else:
            print("✓ AI tracks show SIGNIFICANTLY MORE diversity than human tracks")
            print("  → In OpenL3 embedding space, AI Afrobeats are more VARIABLE.")
    else:
        print("○ No significant difference in diversity between human and AI tracks")
        print("  → In OpenL3 space, both collections have similar internal variety.")

    # ----------------- Visualization -----------------
    print("\n📊 Creating visualizations...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1) Distribution of pairwise distances
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(h_dists, bins=30, alpha=0.6, label="Human", density=True)
    ax1.hist(a_dists, bins=30, alpha=0.6, label="AI", density=True)
    ax1.axvline(human_mean, color="blue", linestyle="--", linewidth=2,
                label=f"Human mean: {human_mean:.2f}")
    ax1.axvline(ai_mean, color="red", linestyle="--", linewidth=2,
                label=f"AI mean: {ai_mean:.2f}")
    ax1.set_xlabel("Pairwise Distance (OpenL3 space)")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of All Pairwise Distances")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Boxplot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    data_for_box = [h_dists, a_dists]
    bp = ax2.boxplot(data_for_box, labels=["Human", "AI"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax2.set_ylabel("Pairwise Distance")
    ax2.set_title("Distance Distributions")
    ax2.grid(axis="y", alpha=0.3)

    # 3) PCA of embedding space (2D)
    ax3 = fig.add_subplot(gs[1, :2])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(all_X_norm)
    human_pca = X_pca[:n_human]
    ai_pca = X_pca[n_human:]

    ax3.scatter(
        human_pca[:, 0],
        human_pca[:, 1],
        alpha=0.6,
        s=80,
        label=f"Human (n={len(human_X)})",
        color="blue",
        edgecolor="black",
    )
    ax3.scatter(
        ai_pca[:, 0],
        ai_pca[:, 1],
        alpha=0.6,
        s=80,
        label=f"AI (n={len(ai_X)})",
        color="red",
        marker="s",
        edgecolor="black",
    )

    plot_cov_ellipse(ax3, human_pca, "blue")
    plot_cov_ellipse(ax3, ai_pca, "red")

    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax3.set_title("OpenL3 Embedding Space (PCA)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4) Dispersion around centroids
    ax4 = fig.add_subplot(gs[1, 2])
    human_centroid_dists = np.linalg.norm(human_X - human_X.mean(axis=0), axis=1)
    ai_centroid_dists = np.linalg.norm(ai_X - ai_X.mean(axis=0), axis=1)

    ax4.boxplot(
        [human_centroid_dists, ai_centroid_dists],
        labels=["Human", "AI"],
        patch_artist=True,
    )
    ax4.set_ylabel("Distance to Group Centroid")
    ax4.set_title("Dispersion Around Group Centroids")
    ax4.grid(axis="y", alpha=0.3)

    # 5) Metric comparison normalized to Human
    ax5 = fig.add_subplot(gs[2, :])

    metrics_to_plot = ["mean_pairwise_dist", "dispersion", "entropy"]
    metric_names = ["Mean Pairwise\nDistance", "Dispersion\n(to centroid)", "Entropy\n(PCA)"]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    human_vals = [human_metrics[m] for m in metrics_to_plot]
    ai_vals = [ai_metrics[m] for m in metrics_to_plot]

    human_norm = [1.0] * len(metrics_to_plot)
    ai_norm = [ai_vals[i] / human_vals[i] for i in range(len(metrics_to_plot))]

    bars1 = ax5.bar(x - width / 2, human_norm, width, label="Human", color="blue", alpha=0.7)
    bars2 = ax5.bar(x + width / 2, ai_norm, width, label="AI", color="red", alpha=0.7)

    ax5.set_ylabel("Relative Value (Human = 1.0)")
    ax5.set_title("Diversity Metrics Comparison (OpenL3, normalized to Human)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_names)
    ax5.legend()
    ax5.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax5.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig_path = out_dir / "overall_diversity_openl3.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {fig_path}")

    # ----------------- Save JSON results -----------------
    results = {
        "human_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in human_metrics.items()
        },
        "ai_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in ai_metrics.items()
        },
        "statistical_test": {
            "test": "Mann-Whitney U",
            "statistic": float(stat),
            "p_value": float(p_val),
            "effect_size_cohens_d": float(effect_size),
            "diversity_ratio": float(diversity_ratio),
        },
        "meta": {
            "human_n": int(len(human_X)),
            "ai_n": int(len(ai_X)),
            "embedding_dim": int(human_X.shape[1]),
            "segment_duration_sec": 30.0,
            "openl3_params": {
                "sr_target": 48000,
                "input_repr": "mel256",
                "content_type": "music",
                "embedding_size": 512,
            },
        },
    }

    json_path = out_dir / "diversity_analysis_openl3.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {json_path}")

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE (OpenL3)")
    print("=" * 72)


if __name__ == "__main__":
    main()
