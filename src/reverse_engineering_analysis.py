"""
Comprehensive Statistical Analysis of AI Music Homogenization
For FAccT Submission - Matched Corpus Design

This script performs rigorous statistical testing of the hypothesis that
AI-generated music exhibits greater acoustic convergence than human music,
controlling for prompt-level similarity through reverse-engineered prompts.

Author: Zoe Slendebroek
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
sns.set_palette("colorblind")

def bh_fdr(pvals):
    """
    Benjamini–Hochberg FDR correction.
    Returns array of adjusted p-values (q-values).
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]

    qvals = np.empty(n, dtype=float)
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked_pvals[i] * n / rank
        q = min(q, prev_q)
        prev_q = q
        qvals[i] = q

    # restore original order
    qvals_corrected = np.empty(n, dtype=float)
    qvals_corrected[order] = qvals
    return qvals_corrected



class MatchedCorpusAnalyzer:
    """
    Comprehensive homogenization analysis for matched AI-Human corpus.
    
    This analyzer implements multiple complementary diagnostics to assess
    whether AI-generated tracks exhibit systematic acoustic convergence
    relative to human-produced music under matched prompting conditions.
    """
    
    def __init__(self, ai_csv, human_csv):
        """
        Initialize analyzer with AI and human feature CSVs.
        
        Parameters
        ----------
        ai_csv : str or Path
            Path to AI-generated tracks feature CSV
        human_csv : str or Path
            Path to human-produced tracks feature CSV
        """
        print("="*80)
        print("MATCHED CORPUS HOMOGENIZATION ANALYZER")
        print("="*80)
        
        # Load data
        self.ai_df = pd.read_csv(ai_csv)
        self.human_df = pd.read_csv(human_csv)
        
        # Add labels
        self.ai_df['label'] = 'AI'
        self.human_df['label'] = 'Human'
        
        # Combine for joint standardization
        self.combined_df = pd.concat([self.ai_df, self.human_df], ignore_index=True)
        
        # Identify feature columns (exclude metadata)
        metadata_cols = ['filename', 'label', 'track_index', 'suno_join_key', 
                        'human_song_file', 'suno_song_name', 'prompt', 
                        'tempo_promptmeta', 'onset_density_promptmeta',
                        'ioi_cv_promptmeta', 'percussive_ratio_promptmeta',
                        'harmonic_ratio_promptmeta', 'repetition_score_promptmeta',
                        'self_similarity_mean_promptmeta', 'rms_mean_promptmeta']
        
        self.feature_cols = [col for col in self.combined_df.columns 
                            if col not in metadata_cols]
        
        print(f"\n✓ Loaded {len(self.ai_df)} AI tracks")
        print(f"✓ Loaded {len(self.human_df)} human tracks")
        print(f"✓ Using {len(self.feature_cols)} acoustic features")
        
        self.combined_df_raw = self.combined_df.copy(deep=True)

        # Standardize features jointly
        self.scaler = StandardScaler()
        self.combined_df[self.feature_cols] = self.scaler.fit_transform(
            self.combined_df[self.feature_cols]
        )
        
        # Split back
        self.ai_features = self.combined_df[self.combined_df['label'] == 'AI'][self.feature_cols]
        self.human_features = self.combined_df[self.combined_df['label'] == 'Human'][self.feature_cols]
        
        # Storage for results
        self.results = {}
        
    
    def test_global_dispersion(self, n_perm=10000, n_boot=10000, random_state=42):
        """
        Reviewer-safe global dispersion test.

        Unit of analysis = tracks (independent).
        Statistic = distance of each track to its corpus centroid in standardized feature space.

        Tests H1: AI has lower dispersion than Human (i.e., smaller distances-to-centroid).
        """
        rng = np.random.default_rng(random_state)

        print("\n" + "="*80)
        print("TEST 1: GLOBAL DISPERSION IN ACOUSTIC FEATURE SPACE (TRACK-LEVEL)")
        print("="*80)
        print("\nH1: AI-generated tracks are less dispersed (smaller distance-to-centroid)")
        print("    than human-produced tracks in standardized feature space.\n")

        # --- compute per-track distances to each group's centroid (independent per track) ---
        ai_X = self.ai_features.values
        hu_X = self.human_features.values

        ai_centroid = ai_X.mean(axis=0)
        hu_centroid = hu_X.mean(axis=0)

        ai_r = np.linalg.norm(ai_X - ai_centroid, axis=1)  # one value per AI track
        hu_r = np.linalg.norm(hu_X - hu_centroid, axis=1)  # one value per Human track

        # --- descriptives ---
        ai_mean, ai_med, ai_std = float(ai_r.mean()), float(np.median(ai_r)), float(ai_r.std(ddof=1))
        hu_mean, hu_med, hu_std = float(hu_r.mean()), float(np.median(hu_r)), float(hu_r.std(ddof=1))

        print(f"AI dist-to-centroid:    μ={ai_mean:.3f}, σ={ai_std:.3f}, median={ai_med:.3f} (n={len(ai_r)})")
        print(f"Human dist-to-centroid: μ={hu_mean:.3f}, σ={hu_std:.3f}, median={hu_med:.3f} (n={len(hu_r)})")
        print(f"Median difference (AI - Human): {ai_med - hu_med:.3f}")

        # --- Mann–Whitney (one-sided): AI < Human ---
        mw_stat, mw_p = stats.mannwhitneyu(ai_r, hu_r, alternative="less")
        print(f"\nMann–Whitney U (one-sided, AI < Human): U={mw_stat:.0f}, p={mw_p:.4g}")

        # --- Effect size: Cliff's delta (robust, nonparametric) ---
        # delta = P(AI > Human) - P(AI < Human). For "AI < Human", delta should be negative.
        # O(n^2) but n is small (~60), so fine.
        greater = 0
        less = 0
        for a in ai_r:
            greater += np.sum(a > hu_r)
            less += np.sum(a < hu_r)
        cliffs_delta = (greater - less) / (len(ai_r) * len(hu_r))
        print(f"Cliff's delta: {cliffs_delta:.3f} (negative => AI more concentrated)")

        # --- Permutation test on mean distance-to-centroid (label-shuffle at track level) ---
        # Recompute group centroids after label shuffle (important).
        all_X = np.vstack([ai_X, hu_X])
        labels = np.array([1]*len(ai_X) + [0]*len(hu_X))  # 1=AI, 0=Human

        def mean_r_for_labels(lbls):
            X_ai = all_X[lbls == 1]
            X_hu = all_X[lbls == 0]
            c_ai = X_ai.mean(axis=0)
            c_hu = X_hu.mean(axis=0)
            r_ai = np.linalg.norm(X_ai - c_ai, axis=1).mean()
            r_hu = np.linalg.norm(X_hu - c_hu, axis=1).mean()
            return r_ai - r_hu  # AI - Human

        observed = mean_r_for_labels(labels)

        perm_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm_lbls = rng.permutation(labels)
            perm_stats[i] = mean_r_for_labels(perm_lbls)

        # one-sided p-value for AI < Human  => (AI-Human) should be small/negative
        perm_p = (np.sum(perm_stats <= observed) + 1) / (n_perm + 1)
        print(f"\nPermutation test (track-level relabel, {n_perm} perms): p={perm_p:.4g}")

        # --- Bootstrap CI for median difference (AI - Human) ---
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            ai_s = rng.choice(ai_r, size=len(ai_r), replace=True)
            hu_s = rng.choice(hu_r, size=len(hu_r), replace=True)
            boot[i] = np.median(ai_s) - np.median(hu_s)

        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        print(f"Bootstrap 95% CI for median(AI)-median(Human): [{ci_lo:.3f}, {ci_hi:.3f}]")

        # --- interpretation (based on permutation p + direction) ---
        if perm_p < 0.001 and (ai_med < hu_med):
            verdict = "✓✓✓ STRONG EVIDENCE"
        elif perm_p < 0.01 and (ai_med < hu_med):
            verdict = "✓✓ MODERATE EVIDENCE"
        elif perm_p < 0.05 and (ai_med < hu_med):
            verdict = "✓ WEAK EVIDENCE"
        else:
            verdict = "✗ NO EVIDENCE"
        # print(f"\n{verdict}: AI tracks are more/less dispersed depending on sign/direction.")
        
        direction = "lower" if ai_med < hu_med else "higher"
        print(f"\n{verdict}: AI dispersion is {direction} than Human (track-to-centroid).")


        self.results['dispersion'] = {
            "ai_mean_dist_to_centroid": ai_mean,
            "ai_median_dist_to_centroid": ai_med,
            "ai_std_dist_to_centroid": ai_std,
            "human_mean_dist_to_centroid": hu_mean,
            "human_median_dist_to_centroid": hu_med,
            "human_std_dist_to_centroid": hu_std,
            "mw_statistic": float(mw_stat),
            "mw_pvalue": float(mw_p),
            "cliffs_delta": float(cliffs_delta),
            "perm_statistic_mean_diff": float(observed),
            "perm_pvalue": float(perm_p),
            "bootstrap_ci_median_diff_lower": float(ci_lo),
            "bootstrap_ci_median_diff_upper": float(ci_hi),
            # keep raw arrays for plotting if you want
            "ai_dist_to_centroid": ai_r,
            "human_dist_to_centroid": hu_r,
        }

        return self.results['dispersion']

    
    def test_feature_variance(self):
        """
        Test H2: AI tracks exhibit lower variance across individual features.
        
        Performs feature-by-feature variance comparison with statistical testing
        to identify which acoustic dimensions show strongest convergence.
        """
        print("\n" + "="*80)
        print("TEST 2: FEATURE-LEVEL VARIANCE STRUCTURE")
        print("="*80)
        print("\nH2: AI-generated tracks exhibit reduced variance across")
        print("    individual acoustic features, indicating contraction along")
        print("    specific acoustic dimensions.\n")
        
        variance_results = []
        
        for feature in self.feature_cols:
            ai_vals = self.ai_features[feature].dropna()
            human_vals = self.human_features[feature].dropna()
            
            ai_var = ai_vals.var()
            human_var = human_vals.var()
            var_ratio = ai_var / human_var if human_var > 0 else np.nan
            
            # Levene's test for equality of variances
            levene_stat, levene_pval = stats.levene(ai_vals, human_vals)
            
            # F-test for variance ratio
            f_stat = ai_var / human_var if human_var > 0 else np.nan
            # For F-test p-value, we use the F distribution
            if not np.isnan(f_stat):
                f_pval = 2 * min(
                    stats.f.cdf(f_stat, len(ai_vals)-1, len(human_vals)-1),
                    1 - stats.f.cdf(f_stat, len(ai_vals)-1, len(human_vals)-1)
                )
            else:
                f_pval = np.nan
            
            variance_results.append({
                'feature': feature,
                'ai_variance': ai_var,
                'human_variance': human_var,
                'variance_ratio': var_ratio,
                'levene_statistic': levene_stat,
                'levene_pvalue': levene_pval,
                'f_statistic': f_stat,
                'f_pvalue': f_pval,
                'significant': levene_pval < 0.05,
                'ai_lower_variance': var_ratio < 1.0
            })
        
        var_df = pd.DataFrame(variance_results)
        var_df = var_df.sort_values('variance_ratio')

        # Benjamini–Hochberg FDR correction (reviewer-safe)
        var_df["levene_p_fdr"] = bh_fdr(var_df["levene_pvalue"].values)
        var_df["significant_fdr"] = var_df["levene_p_fdr"] < 0.05
        
        # Summary statistics
        n_lower = (var_df['variance_ratio'] < 1.0).sum()
        n_sig_lower = ((var_df["variance_ratio"] < 1.0) & 
               (var_df["significant_fdr"])).sum()

        
        mean_ratio = var_df['variance_ratio'].mean()
        median_ratio = var_df['variance_ratio'].median()
        
        print(f"Features with AI variance < Human: {n_lower}/{len(var_df)} ({n_lower/len(var_df)*100:.1f}%)")
        print(f"Statistically significant (BH-FDR q<0.05): {n_sig_lower}/{len(var_df)}")

        
        
        
        print(f"\nMean variance ratio (AI/Human): {mean_ratio:.3f}")
        print(f"Median variance ratio: {median_ratio:.3f}")
        
        # Top contracted features
        print(f"\nTop 15 features with strongest AI convergence:")
        print(
            var_df.head(15)[
                ["feature", "variance_ratio", "levene_pvalue", "levene_p_fdr"]
            ].to_string(index=False)
        )
        # Interpretation
        if n_lower >= len(var_df) * 0.75 and median_ratio < 0.85:
            print("\n✓✓✓ STRONG EVIDENCE: Systematic variance reduction across features")
        elif n_lower >= len(var_df) * 0.60 and median_ratio < 0.90:
            print("\n✓✓ MODERATE EVIDENCE: Substantial variance reduction")
        elif n_lower >= len(var_df) * 0.50:
            print("\n✓ WEAK EVIDENCE: Some variance reduction")
        else:
            print("\n✗ NO EVIDENCE: No systematic variance reduction")
        
        self.results['variance'] = var_df
        return var_df
    
    
    def test_entropy_redundancy(self):
        """
        Test H3: AI tracks exhibit lower entropy (higher redundancy).
        
        Measures informational diversity through histogram entropy
        for each feature distribution.
        """
        print("\n" + "="*80)
        print("TEST 3: INFORMATIONAL REDUNDANCY AND ENTROPY")
        print("="*80)
        print("\nH3: AI-generated tracks exhibit lower distributional entropy,")
        print("    indicating greater predictability and redundancy.\n")
        
        entropy_results = []
        n_bins = 20  # Consistent binning
        
        for feature in self.feature_cols:
            ai_vals = self.ai_features[feature].dropna()
            human_vals = self.human_features[feature].dropna()
            
            # Create histograms with same bins
            combined = np.concatenate([ai_vals, human_vals])
            bin_edges = np.histogram_bin_edges(combined, bins=n_bins)
            
            ai_hist, _ = np.histogram(ai_vals, bins=bin_edges, density=True)
            human_hist, _ = np.histogram(human_vals, bins=bin_edges, density=True)
            
            # Normalize to probability distributions
            ai_hist = ai_hist / ai_hist.sum() if ai_hist.sum() > 0 else ai_hist
            human_hist = human_hist / human_hist.sum() if human_hist.sum() > 0 else human_hist
            
            # Add small constant to avoid log(0)
            epsilon = 1e-10
            ai_hist = ai_hist + epsilon
            human_hist = human_hist + epsilon
            
            # Shannon entropy
            ai_entropy = -np.sum(ai_hist * np.log2(ai_hist))
            human_entropy = -np.sum(human_hist * np.log2(human_hist))
            
            entropy_ratio = ai_entropy / human_entropy if human_entropy > 0 else np.nan
            
            entropy_results.append({
                'feature': feature,
                'ai_entropy': ai_entropy,
                'human_entropy': human_entropy,
                'entropy_ratio': entropy_ratio,
                'ai_lower_entropy': entropy_ratio < 1.0
            })
        
        entropy_df = pd.DataFrame(entropy_results)
        entropy_df = entropy_df.sort_values('entropy_ratio')
        
        n_lower = (entropy_df['entropy_ratio'] < 1.0).sum()
        mean_ratio = entropy_df['entropy_ratio'].mean()
        median_ratio = entropy_df['entropy_ratio'].median()
        
        print(f"Features with AI entropy < Human: {n_lower}/{len(entropy_df)} ({n_lower/len(entropy_df)*100:.1f}%)")
        print(f"\nMean entropy ratio (AI/Human): {mean_ratio:.3f}")
        print(f"Median entropy ratio: {median_ratio:.3f}")
        
        print(f"\nTop 10 features with lowest AI entropy:")
        print(entropy_df.head(10)[['feature', 'entropy_ratio']].to_string(index=False))
        
        # Interpretation
        if n_lower >= len(entropy_df) * 0.70 and median_ratio < 0.90:
            print("\n✓✓✓ STRONG EVIDENCE: Systematic entropy reduction")
        elif n_lower >= len(entropy_df) * 0.60 and median_ratio < 0.95:
            print("\n✓✓ MODERATE EVIDENCE: Substantial entropy reduction")
        elif n_lower >= len(entropy_df) * 0.50:
            print("\n✓ WEAK EVIDENCE: Some entropy reduction")
        else:
            print("\n✗ NO EVIDENCE: No systematic entropy reduction")
        
        self.results['entropy'] = entropy_df
        return entropy_df
    
    
    def test_geometric_coverage(self):
        """
        Test H4: AI tracks cover less geometric volume in feature space.
        
        Uses PCA projection and convex hull volume to assess spatial coverage.
        """
        print("\n" + "="*80)
        print("TEST 4: GEOMETRIC COVERAGE OF FEATURE SPACE")
        print("="*80)
        print("\nH4: AI-generated tracks occupy a smaller geometric region")
        print("    of acoustic feature space.\n")
        
        # PCA for dimensionality reduction
        n_components = min(10, len(self.feature_cols), len(self.ai_features))
        pca = PCA(n_components=n_components)
        
        X_all = self.combined_df[self.feature_cols].values
        Z_all = pca.fit_transform(X_all)

        ai_pca = Z_all[self.combined_df['label'].values == 'AI']
        human_pca = Z_all[self.combined_df['label'].values == 'Human']

        
        print(f"PCA with {n_components} components")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        results = {}
        
        # 1. Range-based coverage
        range_ratios = []
        for i in range(n_components):
            ai_range = ai_pca[:, i].max() - ai_pca[:, i].min()
            human_range = human_pca[:, i].max() - human_pca[:, i].min()
            range_ratio = ai_range / human_range if human_range > 0 else np.nan
            range_ratios.append(range_ratio)
        
        results['mean_range_ratio'] = np.nanmean(range_ratios)
        results['median_range_ratio'] = np.nanmedian(range_ratios)
        
        print(f"\nRange-based coverage:")
        print(f"  Mean range ratio (AI/Human): {results['mean_range_ratio']:.3f}")
        print(f"  Components where AI < Human: {sum(r < 1 for r in range_ratios)}/{n_components}")
        
        # 2. Convex hull volume
        try:
            # Use first 3-5 components for hull
            n_hull_dims = min(5, n_components)
            ai_hull = ConvexHull(ai_pca[:, :n_hull_dims])
            human_hull = ConvexHull(human_pca[:, :n_hull_dims])
            
            results['ai_hull_volume'] = ai_hull.volume
            results['human_hull_volume'] = human_hull.volume
            results['hull_volume_ratio'] = ai_hull.volume / human_hull.volume
            
            print(f"\nConvex hull volume ({n_hull_dims}D):")
            print(f"  AI: {ai_hull.volume:.3e}")
            print(f"  Human: {human_hull.volume:.3e}")
            print(f"  Ratio (AI/Human): {results['hull_volume_ratio']:.3f}")
        except Exception as e:
            print(f"\nConvex hull computation failed: {e}")
            results['hull_volume_ratio'] = np.nan
        
        # 3. Effective dimensionality (participation ratio)
        ai_cov = np.cov(self.ai_features.values.T)
        human_cov = np.cov(self.human_features.values.T)
        
        ai_eigvals = np.linalg.eigvalsh(ai_cov)
        human_eigvals = np.linalg.eigvalsh(human_cov)
        
        # Remove negative eigenvalues (numerical artifacts)
        ai_eigvals = ai_eigvals[ai_eigvals > 0]
        human_eigvals = human_eigvals[human_eigvals > 0]
        
        ai_pr = (np.sum(ai_eigvals)**2) / np.sum(ai_eigvals**2)
        human_pr = (np.sum(human_eigvals)**2) / np.sum(human_eigvals**2)
        
        results['ai_participation_ratio'] = ai_pr
        results['human_participation_ratio'] = human_pr
        
        print(f"\nEffective dimensionality (participation ratio):")
        print(f"  AI: {ai_pr:.2f}")
        print(f"  Human: {human_pr:.2f}")
        
        # Store PCA for visualization
        results['pca'] = pca
        results['ai_pca'] = ai_pca
        results['human_pca'] = human_pca
        
        # Interpretation
        coverage_score = 0
        if results['mean_range_ratio'] < 0.90:
            coverage_score += 1
        if 'hull_volume_ratio' in results and results['hull_volume_ratio'] < 0.75:
            coverage_score += 1
        if ai_pr < human_pr * 0.90:
            coverage_score += 1
        
        if coverage_score >= 2:
            print("\n✓✓ STRONG EVIDENCE: AI tracks cover less feature space")
        elif coverage_score == 1:
            print("\n✓ MODERATE EVIDENCE: Some coverage reduction")
        else:
            print("\n✗ WEAK EVIDENCE: Minimal coverage difference")
        
        self.results['coverage'] = results
        return results
    
    
    def test_classification_separability(self):
        """
        Test H5: AI and human tracks are separable by acoustic features alone.
        
        Uses supervised classification as consistency check. High accuracy
        confirms systematic acoustic differences.
        """
        print("\n" + "="*80)
        print("TEST 5: SUPERVISED CLASSIFICATION (CONSISTENCY CHECK)")
        print("="*80)
        print("\nH5: AI and human tracks exhibit systematic acoustic differences")
        print("    that enable classification with high accuracy.\n")
        
        # Prepare RAW (unscaled) data to avoid leakage
        X = self.combined_df_raw[self.feature_cols].values
        y = (self.combined_df_raw['label'] == 'AI').astype(int).values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
        }

        results = []

        print("Cross-validated classification performance:\n")
        print(f"{'Classifier':<25} {'Accuracy':<15} {'AUC':<15}")
        print("-" * 55)

        for name, clf in classifiers.items():
            # Scale *within each fold* (no test-fold leakage)
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])

            scores = cross_validate(
                model, X, y, cv=cv,
                scoring=['accuracy', 'roc_auc'],
                return_train_score=False
            )

            acc_mean = scores['test_accuracy'].mean()
            acc_std = scores['test_accuracy'].std()
            auc_mean = scores['test_roc_auc'].mean()
            auc_std = scores['test_roc_auc'].std()

            print(f"{name:<25} {acc_mean:.3f} ± {acc_std:.3f}   {auc_mean:.3f} ± {auc_std:.3f}")

            results.append({
                'classifier': name,
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'auc_mean': auc_mean,
                'auc_std': auc_std
            })

        # Feature importance model (descriptive): train on ALL data, but do it cleanly too
        print("\n\nTraining Random Forest for feature importance analysis...")
        rf_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        rf_model.fit(X, y)

        # Pull importances from the RF inside the pipeline
        rf = rf_model.named_steps['clf']
        importances = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 discriminative features:")
        print(importances.head(20).to_string(index=False))

        best_auc = max([r['auc_mean'] for r in results])
        
        if best_auc > 0.95:
            print("\n✓✓✓ STRONG SEPARABILITY: AI and human tracks are highly distinguishable")
        elif best_auc > 0.85:
            print("\n✓✓ MODERATE SEPARABILITY: Clear acoustic differences")
        elif best_auc > 0.70:
            print("\n✓ WEAK SEPARABILITY: Some differences detectable")
        else:
            print("\n✗ NO SEPARABILITY: Cannot distinguish AI from human")
        
        self.results['classification'] = {
            'results': pd.DataFrame(results),
            'feature_importance': importances,
            'best_auc': best_auc
        }
        
        return self.results['classification']
    
    
    def generate_visualizations(self, output_dir='figures'):
        """
        Generate publication-quality figures for paper.
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Figure 1: Composite overview (4-panel)
        self._plot_composite_figure(output_path)
        
        # Figure 2: Pairwise distance distributions
        self._plot_distance_distributions(output_path)
        
        # Figure 3: Feature variance ratios
        self._plot_variance_analysis(output_path)
        
        # Figure 4: PCA projection with density
        self._plot_pca_projection(output_path)
        
        # Figure 5: Feature importance from classifier
        self._plot_feature_importance(output_path)
        
        print(f"\n✓ All figures saved to {output_path}/")
    
    
    def _plot_composite_figure(self, output_path):
        """Create 4-panel composite figure (like Figure 2 in paper)"""
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: PCA projection
        ax1 = fig.add_subplot(gs[0, 0])
        pca_data = self.results['coverage']
        ai_pca = pca_data['ai_pca']
        human_pca = pca_data['human_pca']
        pca = pca_data['pca']
        
        ax1.scatter(human_pca[:, 0], human_pca[:, 1], 
                   alpha=0.6, s=50, label='Human', color='#1f77b4')
        ax1.scatter(ai_pca[:, 0], ai_pca[:, 1], 
                   alpha=0.6, s=50, label='AI', color='#ff7f0e')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.legend()
        ax1.set_title('(A) PCA Projection of Acoustic Features', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Panel B: Distance-to-centroid distributions (track-level, independent)
        ax2 = fig.add_subplot(gs[0, 1])
        disp = self.results['dispersion']

        ai_d = disp['ai_dist_to_centroid']
        hu_d = disp['human_dist_to_centroid']

        bins = np.linspace(min(ai_d.min(), hu_d.min()), max(ai_d.max(), hu_d.max()), 30)

        ax2.hist(hu_d, bins=bins, alpha=0.6, label='Human', density=True)
        ax2.hist(ai_d, bins=bins, alpha=0.6, label='AI', density=True)

        ax2.axvline(np.mean(hu_d), linestyle="--", linewidth=1)
        ax2.axvline(np.mean(ai_d), linestyle="--", linewidth=1)

        ax2.set_xlabel('Distance to corpus centroid')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.set_title('(B) Distance-to-Centroid Distributions', fontweight='bold')
        ax2.grid(alpha=0.3)

        
        # Panel C: Variance ratios
        ax3 = fig.add_subplot(gs[1, 0])
        var_df = self.results['variance'].copy()
        
        # Sort and plot
        var_df = var_df.sort_values('variance_ratio')
        colors = ['#d62728' if r < 1.0 else '#2ca02c' for r in var_df['variance_ratio']]
        
        y_pos = np.arange(len(var_df))
        ax3.barh(y_pos, var_df['variance_ratio'], color=colors, alpha=0.7)
        ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Variance Ratio (AI/Human)')
        ax3.set_ylabel('Feature Index')
        ax3.set_title('(C) Feature-Level Variance Ratios', fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
        ax3.set_ylim(-1, len(var_df))
        ax3.set_yticks([])  # Too many to label
        
        # Panel D: Feature importance
        ax4 = fig.add_subplot(gs[1, 1])
        importances = self.results['classification']['feature_importance'].head(15)

        ax4.barh(range(len(importances)), importances['importance'], color='#9467bd')
        ax4.set_yticks(range(len(importances)))
        ax4.set_yticklabels(importances['feature'], fontsize=8)
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('(D) Top Discriminative Features', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(alpha=0.3, axis='x')


        # Overall figure title + save
        fig.suptitle('AI Music Homogenization Diagnostics (Matched Corpus)', fontweight='bold', y=0.98)
        out_file = output_path / "figure1_composite.png"
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

    def _plot_distance_distributions(self, output_path):
        """Figure: Pairwise distance distributions + summary stats"""
        disp = self.results['dispersion']
        ai_d = disp['ai_dist_to_centroid']
        hu_d = disp['human_dist_to_centroid']

        fig = plt.figure(figsize=(6.5, 4.0))
        bins = np.linspace(min(ai_d.min(), hu_d.min()), max(ai_d.max(), hu_d.max()), 40)

        plt.hist(hu_d, bins=bins, alpha=0.6, density=True, label="Human")
        plt.hist(ai_d, bins=bins, alpha=0.6, density=True, label="AI")

        plt.axvline(np.mean(hu_d), linestyle="--", linewidth=1)
        plt.axvline(np.mean(ai_d), linestyle="--", linewidth=1)

        plt.xlabel("Distance to corpus centroid (standardized feature space)")
        plt.ylabel("Density")
        plt.title("Within-corpus dispersion (track-level distance to centroid)")
        plt.legend()
        plt.grid(alpha=0.3)

        out_file = output_path / "figure2_distance_distributions.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

    def _plot_variance_analysis(self, output_path):
        """Figure: Variance ratios (AI/Human) across features, highlight strongest contractions."""
        var_df = self.results['variance'].copy().sort_values("variance_ratio")
        fig = plt.figure(figsize=(7.0, 6.5))

        colors = ['#d62728' if r < 1.0 else '#2ca02c' for r in var_df['variance_ratio']]
        y_pos = np.arange(len(var_df))

        plt.barh(y_pos, var_df["variance_ratio"], alpha=0.75, color=colors)
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1)

        plt.yticks([])  # too many labels
        plt.xlabel("Variance ratio (AI / Human)")
        plt.title("Feature-level variance ratios")
        plt.grid(alpha=0.3, axis="x")

        out_file = output_path / "figure3_variance_ratios.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

        # Also save a labeled top-20 plot for readability
        topk = 20
        top = var_df.head(topk).copy()
        fig2 = plt.figure(figsize=(7.0, 5.5))
        plt.barh(range(len(top)), top["variance_ratio"], alpha=0.8)
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
        plt.yticks(range(len(top)), top["feature"], fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Variance ratio (AI / Human)")
        plt.title(f"Top {topk} most contracted features (AI < Human)")
        plt.grid(alpha=0.3, axis="x")

        out_file2 = output_path / "figure3b_top20_variance_ratios_labeled.png"
        plt.savefig(out_file2, bbox_inches="tight")
        plt.close(fig2)

    def _plot_pca_projection(self, output_path):
        """Figure: PCA 2D projection with marginal density / scatter."""
        cov = self.results['coverage']
        ai_pca = cov['ai_pca']
        hu_pca = cov['human_pca']
        pca = cov['pca']

        fig = plt.figure(figsize=(6.5, 5.5))
        plt.scatter(hu_pca[:, 0], hu_pca[:, 1], alpha=0.6, s=45, label="Human")
        plt.scatter(ai_pca[:, 0], ai_pca[:, 1], alpha=0.6, s=45, label="AI")

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title("PCA projection of standardized acoustic features")
        plt.legend()
        plt.grid(alpha=0.3)

        out_file = output_path / "figure4_pca_projection.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

    def _plot_feature_importance(self, output_path):
        """Figure: Feature importance from the Random Forest classifier."""
        importances = self.results['classification']['feature_importance'].head(25).copy()

        fig = plt.figure(figsize=(7.0, 6.0))
        plt.barh(range(len(importances)), importances["importance"], alpha=0.85)
        plt.yticks(range(len(importances)), importances["feature"], fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Random Forest feature importance")
        plt.title("Top discriminative features (AI vs Human)")
        plt.grid(alpha=0.3, axis="x")

        out_file = output_path / "figure5_feature_importance.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

    def save_all_results(self, output_prefix="matched_homogenization"):
        """Save main result tables to CSV for paper/supplement."""
        out = Path("results")
        out.mkdir(exist_ok=True)

        if 'variance' in self.results and isinstance(self.results['variance'], pd.DataFrame):
            self.results['variance'].to_csv(out / f"{output_prefix}_variance.csv", index=False)

        if 'entropy' in self.results and isinstance(self.results['entropy'], pd.DataFrame):
            self.results['entropy'].to_csv(out / f"{output_prefix}_entropy.csv", index=False)

        if 'classification' in self.results:
            self.results['classification']['results'].to_csv(out / f"{output_prefix}_classification_cv.csv", index=False)
            self.results['classification']['feature_importance'].to_csv(out / f"{output_prefix}_feature_importance.csv", index=False)

        if 'dispersion' in self.results:
            # save summary only (not raw distance arrays)
            disp = self.results['dispersion'].copy()
            disp.pop('ai_dist_to_centroid', None)
            disp.pop('human_dist_to_centroid', None)
            pd.DataFrame([disp]).to_csv(out / f"{output_prefix}_dispersion_summary.csv", index=False)

        if 'coverage' in self.results:
            cov = self.results['coverage'].copy()
            # remove objects/arrays not CSV-friendly
            cov.pop('pca', None)
            cov.pop('ai_pca', None)
            cov.pop('human_pca', None)
            pd.DataFrame([cov]).to_csv(out / f"{output_prefix}_coverage_summary.csv", index=False)

        print(f"\n✓ Results saved to {out}/ with prefix '{output_prefix}_'")

# ----------------------------
# Run everything
# ----------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    AI_CSV = project_root / "data" / "afrobeat" / "ai_tracks_features_67_with_prompts.csv"
    HUMAN_CSV = project_root / "data" / "afrobeat" / "human_preview_features_67.csv"

    analyzer = MatchedCorpusAnalyzer(AI_CSV, HUMAN_CSV)

    analyzer.test_global_dispersion()
    analyzer.test_feature_variance()
    analyzer.test_entropy_redundancy()
    analyzer.test_geometric_coverage()
    analyzer.test_classification_separability()

    analyzer.generate_visualizations(output_dir="figures")
    analyzer.save_all_results(output_prefix="matched_homogenization")
