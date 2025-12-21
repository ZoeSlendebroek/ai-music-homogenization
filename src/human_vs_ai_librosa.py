#!/usr/bin/env python3
"""
This script uses librosa to explore and measure the OVERALL DIVERSITY within each collection:
- How spread out are human tracks in feature space?
- How spread out are AI tracks in feature space?
- Which collection is more homogeneous?
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def extract_features(filepath, sr=22050):
    """Extract features consistently."""
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    
    # Middle 30s
    target_len = int(30 * sr)
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = librosa.util.fix_length(y, size=target_len)
    
    # Features
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    
    features = {
        'tempo': float(tempo),
        'spectral_centroid': float(spec_cent),
        'spectral_bandwidth': float(spec_band),
        'spectral_rolloff': float(spec_roll),
        'zero_crossing_rate': float(zcr),
    }
    
    for i, v in enumerate(mfcc, start=1):
        features[f'mfcc_{i}'] = float(v)
    for i, v in enumerate(chroma, start=1):
        features[f'chroma_{i}'] = float(v)
    
    return features


def compute_diversity_metrics(X_norm, label):
    """
    Compute multiple diversity metrics for a collection.
    """
    n = len(X_norm)
    
    # 1. Mean pairwise distance
    distances = pdist(X_norm, metric='euclidean')
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    # 2. Centroid-based dispersion (average distance to centroid)
    centroid = X_norm.mean(axis=0)
    centroid_dists = np.linalg.norm(X_norm - centroid, axis=1)
    dispersion = centroid_dists.mean()
    
    # 3. Volume (determinant of covariance matrix)
    # Measures "spread" in feature space
    cov_matrix = np.cov(X_norm.T)
    # Use log determinant to avoid numerical issues
    sign, logdet = np.linalg.slogdet(cov_matrix)
    volume = logdet if sign > 0 else -np.inf
    
    # 4. Entropy (in PCA space)
    pca = PCA()
    pca.fit(X_norm)
    # Entropy of explained variance
    var_ratios = pca.explained_variance_ratio_
    entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-10))
    
    print(f"\n{label}:")
    print(f"  Mean pairwise distance: {mean_dist:.3f} ± {std_dist:.3f}")
    print(f"  Dispersion (dist to centroid): {dispersion:.3f}")
    print(f"  Log-volume (covariance det): {volume:.3f}")
    print(f"  Entropy (PCA): {entropy:.3f}")
    print(f"  Number of tracks: {n}")
    
    return {
        'mean_pairwise_dist': mean_dist,
        'std_pairwise_dist': std_dist,
        'dispersion': dispersion,
        'volume': volume,
        'entropy': entropy,
        'n_tracks': n,
        'all_distances': distances
    }


def main():
    print("="*70)
    print("OVERALL DIVERSITY: HUMAN vs AI AFROBEATS")
    print("="*70)
    print("Research Question: Which collection has more internal variety?")
    
    # Load metadata
    human_meta = pd.read_csv("../data/human/afrobeat/meta.csv")
    ai_meta = pd.read_csv("../data/ai/afrobeat/meta.csv")
    
    ai_audio_dir = Path("../data/ai/afrobeat/audio")
    
    # Check if human has features already
    feature_cols = ['tempo', 'spectral_centroid', 'spectral_bandwidth', 
                   'spectral_rolloff', 'zero_crossing_rate'] + \
                   [f'mfcc_{i}' for i in range(1, 14)] + \
                   [f'chroma_{i}' for i in range(1, 13)]
    
    if all(col in human_meta.columns for col in feature_cols):
        print("\n✓ Using pre-extracted human features from CSV")
        human_feat_df = human_meta[['filename'] + feature_cols].copy()
        human_feat_df['source'] = 'Human'
    else:
        print("\n⚠️  Need to extract human features from audio files")
        return
    
    # Extract AI features
    print("\nExtracting AI features...")
    ai_features = []
    for idx, row in ai_meta.iterrows():
        filepath = ai_audio_dir / row['filename']
        if not filepath.exists():
            continue
        print(f"  [{idx+1:2d}/20] {row['filename']}...", end=" ")
        feat = extract_features(filepath)
        feat['filename'] = row['filename']
        feat['source'] = 'AI'
        ai_features.append(feat)
        print("✓")
    
    ai_feat_df = pd.DataFrame(ai_features)
    
    # Combine
    all_data = pd.concat([human_feat_df, ai_feat_df], ignore_index=True)
    
    print(f"\nComparing {len(human_feat_df)} human tracks vs {len(ai_feat_df)} AI tracks")
    print(f"   Using {len(feature_cols)} features")
    
    # Standardize features JOINTLY
    X = all_data[feature_cols].values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    human_X = X_norm[:len(human_feat_df)]
    ai_X = X_norm[len(human_feat_df):]
    
    # Compute diversity metrics
    print("\n" + "="*70)
    print("DIVERSITY METRICS")
    print("="*70)
    
    human_metrics = compute_diversity_metrics(human_X, "HUMAN TRACKS")
    ai_metrics = compute_diversity_metrics(ai_X, "AI TRACKS")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    # Permutation test for mean pairwise distance
    h_dists = human_metrics['all_distances']
    a_dists = ai_metrics['all_distances']
    
    # Two-sample comparison
    from scipy.stats import mannwhitneyu
    stat, p_val = mannwhitneyu(h_dists, a_dists, alternative='two-sided')
    
    print(f"\nMann-Whitney U test (pairwise distances):")
    print(f"  Human mean: {h_dists.mean():.3f}")
    print(f"  AI mean: {a_dists.mean():.3f}")
    print(f"  Difference: {h_dists.mean() - a_dists.mean():.3f}")
    print(f"  U-statistic: {stat:.1f}")
    print(f"  p-value: {p_val:.4f}")
    
    # Effect size
    effect_size = (h_dists.mean() - a_dists.mean()) / np.sqrt((h_dists.std()**2 + a_dists.std()**2) / 2)
    print(f"  Cohen's d: {effect_size:.3f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if p_val < 0.05:
        if h_dists.mean() > a_dists.mean():
            print("✓ Human tracks show SIGNIFICANTLY MORE diversity than AI tracks")
            print("  → AI-generated Afrobeats are more HOMOGENEOUS")
        else:
            print("✓ AI tracks show SIGNIFICANTLY MORE diversity than human tracks")
            print("  → AI-generated Afrobeats are more VARIABLE")
    else:
        print("○ No significant difference in diversity between human and AI tracks")
        print(f"  → Both collections have similar internal variety")
    
    diversity_ratio = h_dists.mean() / a_dists.mean()
    print(f"\nDiversity ratio (Human/AI): {diversity_ratio:.2f}x")
    
    if diversity_ratio > 1.1:
        print(f"  Human tracks are {(diversity_ratio-1)*100:.1f}% more diverse")
    elif diversity_ratio < 0.9:
        print(f"  AI tracks are {(1/diversity_ratio-1)*100:.1f}% more diverse")
    else:
        print(f"  Very similar diversity levels")
    

    # VISUALISATIONS
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distribution of pairwise distances
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(h_dists, bins=30, alpha=0.6, label='Human', density=True, color='blue')
    ax1.hist(a_dists, bins=30, alpha=0.6, label='AI', density=True, color='red')
    ax1.axvline(h_dists.mean(), color='blue', linestyle='--', linewidth=2, label=f'Human mean: {h_dists.mean():.2f}')
    ax1.axvline(a_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'AI mean: {a_dists.mean():.2f}')
    ax1.set_xlabel('Pairwise Distance')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of All Pairwise Distances')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    data_for_box = [h_dists, a_dists]
    bp = ax2.boxplot(data_for_box, labels=['Human', 'AI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Pairwise Distance')
    ax2.set_title('Distance Distributions')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. PCA visualization
    ax3 = fig.add_subplot(gs[1, :2])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)
    
    human_pca = X_pca[:len(human_feat_df)]
    ai_pca = X_pca[len(human_feat_df):]
    
    ax3.scatter(human_pca[:, 0], human_pca[:, 1], alpha=0.6, s=100, 
               label=f'Human (n={len(human_feat_df)})', color='blue', edgecolor='black')
    ax3.scatter(ai_pca[:, 0], ai_pca[:, 1], alpha=0.6, s=100,
               label=f'AI (n={len(ai_feat_df)})', color='red', marker='s', edgecolor='black')
    
    # Add ellipses showing spread
    from matplotlib.patches import Ellipse
    
    def plot_cov_ellipse(ax, data, color, alpha=0.3):
        mean = data.mean(axis=0)
        cov = np.cov(data.T)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=mean, width=lambda_[0]*4, height=lambda_[1]*4,
                     angle=np.rad2deg(np.arccos(v[0, 0])),
                     facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
        ax.add_patch(ell)
    
    plot_cov_ellipse(ax3, human_pca, 'blue')
    plot_cov_ellipse(ax3, ai_pca, 'red')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax3.set_title('Feature Space Distribution (PCA)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Centroid distances
    ax4 = fig.add_subplot(gs[1, 2])
    human_centroid_dists = np.linalg.norm(human_X - human_X.mean(axis=0), axis=1)
    ai_centroid_dists = np.linalg.norm(ai_X - ai_X.mean(axis=0), axis=1)
    
    ax4.boxplot([human_centroid_dists, ai_centroid_dists], 
               labels=['Human', 'AI'], patch_artist=True)
    ax4.set_ylabel('Distance to Centroid')
    ax4.set_title('Dispersion Around Centroid')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Metric comparison bars
    ax5 = fig.add_subplot(gs[2, :])
    
    metrics_to_plot = ['mean_pairwise_dist', 'dispersion', 'entropy']
    metric_names = ['Mean Pairwise\nDistance', 'Dispersion\n(to centroid)', 'Entropy\n(PCA)']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    human_vals = [human_metrics[m] for m in metrics_to_plot]
    ai_vals = [ai_metrics[m] for m in metrics_to_plot]
    
    # Normalize to human values for easier comparison
    human_norm = [1.0] * len(metrics_to_plot)
    ai_norm = [ai_vals[i] / human_vals[i] for i in range(len(metrics_to_plot))]
    
    bars1 = ax5.bar(x - width/2, human_norm, width, label='Human', color='blue', alpha=0.7)
    bars2 = ax5.bar(x + width/2, ai_norm, width, label='AI', color='red', alpha=0.7)
    
    ax5.set_ylabel('Relative Value (Human = 1.0)')
    ax5.set_title('Diversity Metrics Comparison (Normalized to Human)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_names)
    ax5.legend()
    ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.savefig('../results/similarity_analysis/overall_diversity_comparison.png', 
               dpi=150, bbox_inches='tight')
    print("Saved visualization to overall_diversity_comparison.png")
    
    # Save detailed results
    results = {
        'human_metrics': human_metrics,
        'ai_metrics': ai_metrics,
        'statistical_test': {
            'test': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p_val),
            'effect_size_cohens_d': float(effect_size),
            'diversity_ratio': float(diversity_ratio)
        }
    }
    
    import json
    with open('../results/similarity_analysis/diversity_analysis.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'human_metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                             for k, v in human_metrics.items()},
            'ai_metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in ai_metrics.items()},
            'statistical_test': results['statistical_test']
        }
        json.dump(results_json, f, indent=2)
    
    print("Saved detailed results to diversity_analysis.json")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()