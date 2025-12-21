"""
Statistical Analysis of AI Music Homogenization
Tests variance, diversity, and feature space coverage between AI and human tracks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class HomogenizationAnalyzer:
    """Comprehensive analysis of musical homogenization"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        
        # side column as key instead of 'label'
        self.ai_df = features_df[features_df['side'] == 'AI'].copy()
        self.human_df = features_df[features_df['side'] == 'Human'].copy()
        
        # excluding new metadata columns
        self.feature_cols = [col for col in features_df.columns 
                            if col not in ['filename', 'side', 'pair_key', 'track_id']]
        
        print(f"Loaded {len(self.ai_df)} AI tracks and {len(self.human_df)} Human tracks")
        print(f"Using {len(self.feature_cols)} features for analysis")
        
        # Standardize features
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        self.ai_df[self.feature_cols] = self.scaler.transform(self.ai_df[self.feature_cols])
        self.human_df[self.feature_cols] = self.scaler.transform(self.human_df[self.feature_cols])
        
        self.results = {}
    
    def analyze_variance_homogeneity(self):
        """Test H1: AI tracks have lower variance (more homogeneous)"""
        print("\n" + "="*60)
        print("VARIANCE HOMOGENEITY ANALYSIS")
        print("="*60)
        
        variance_results = []
        
        for feature in self.feature_cols:
            ai_var = self.ai_df[feature].var()
            human_var = self.human_df[feature].var()
            
            # Levene's test for equality of variances
            stat, p_value = stats.levene(
                self.ai_df[feature].dropna(), 
                self.human_df[feature].dropna()
            )
            
            # Effect size: variance ratio
            var_ratio = ai_var / human_var if human_var > 0 else np.nan
            
            variance_results.append({
                'feature': feature,
                'ai_variance': ai_var,
                'human_variance': human_var,
                'variance_ratio': var_ratio,
                'levene_statistic': stat,
                'levene_pvalue': p_value,
                'significant': p_value < 0.05,
                'ai_more_homogeneous': var_ratio < 1.0
            })
        
        var_df = pd.DataFrame(variance_results)
        var_df = var_df.sort_values('variance_ratio')
        
        # Summary statistics
        print(f"\nFeatures where AI is more homogeneous (var_ratio < 1): "
              f"{(var_df['variance_ratio'] < 1).sum()} / {len(var_df)}")
        print(f"Statistically significant differences (p < 0.05): "
              f"{var_df['significant'].sum()} / {len(var_df)}")
        
        print(f"\nMean variance ratio (AI/Human): {var_df['variance_ratio'].mean():.3f}")
        print(f"Median variance ratio: {var_df['variance_ratio'].median():.3f}")
        
        # Most homogeneous features
        print("\nTop 10 features where AI is most homogeneous:")
        print(var_df.head(10)[['feature', 'variance_ratio', 'levene_pvalue']])
        
        self.results['variance'] = var_df
        return var_df
    
    def analyze_pairwise_distances(self):
        """Test H2: AI tracks are more similar to each other (tighter clustering)"""
        print("\n" + "="*60)
        print("PAIRWISE DISTANCE ANALYSIS")
        print("="*60)
        
        # Compute pairwise distances
        ai_features = self.ai_df[self.feature_cols].values
        human_features = self.human_df[self.feature_cols].values
        
        ai_distances = pdist(ai_features, metric='euclidean')
        human_distances = pdist(human_features, metric='euclidean')
        
        # Statistical tests
        ks_stat, ks_pvalue = stats.ks_2samp(ai_distances, human_distances)
        mw_stat, mw_pvalue = stats.mannwhitneyu(ai_distances, human_distances, alternative='less')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(ai_distances)**2 + np.std(human_distances)**2) / 2)
        cohens_d = (np.mean(ai_distances) - np.mean(human_distances)) / pooled_std
        
        results = {
            'ai_mean_distance': np.mean(ai_distances),
            'ai_median_distance': np.median(ai_distances),
            'ai_std_distance': np.std(ai_distances),
            'human_mean_distance': np.mean(human_distances),
            'human_median_distance': np.median(human_distances),
            'human_std_distance': np.std(human_distances),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mannwhitney_statistic': mw_stat,
            'mannwhitney_pvalue': mw_pvalue,
            'cohens_d': cohens_d
        }
        
        print(f"\nAI mean pairwise distance: {results['ai_mean_distance']:.3f} ± {results['ai_std_distance']:.3f}")
        print(f"Human mean pairwise distance: {results['human_mean_distance']:.3f} ± {results['human_std_distance']:.3f}")
        print(f"\nDifference: {results['ai_mean_distance'] - results['human_mean_distance']:.3f}")
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"Mann-Whitney U test (H1: AI < Human): p = {mw_pvalue:.4f}")
        
        if cohens_d < -0.2:
            print("\n✓ AI tracks show tighter clustering (small-to-large effect)")
        
        self.results['distances'] = results
        self.results['ai_distances'] = ai_distances
        self.results['human_distances'] = human_distances
        
        return results
    
    def analyze_feature_space_coverage(self):
        """Test H3: AI tracks cover less of the feature space"""
        print("\n" + "="*60)
        print("FEATURE SPACE COVERAGE ANALYSIS")
        print("="*60)
        
        results = {}
        
        # 1. Feature range analysis
        range_ratios = []
        for feature in self.feature_cols:
            ai_range = self.ai_df[feature].max() - self.ai_df[feature].min()
            human_range = self.human_df[feature].max() - self.human_df[feature].min()
            range_ratio = ai_range / human_range if human_range > 0 else np.nan
            range_ratios.append(range_ratio)
        
        results['mean_range_ratio'] = np.nanmean(range_ratios)
        results['median_range_ratio'] = np.nanmedian(range_ratios)
        
        print(f"\nMean feature range ratio (AI/Human): {results['mean_range_ratio']:.3f}")
        print(f"Features where AI covers less range: {sum(r < 1 for r in range_ratios if not np.isnan(r))} / {len(range_ratios)}")
        
        # 2. Convex hull volume (in PCA space)
        pca = PCA(n_components=min(10, len(self.feature_cols)))
        ai_pca = pca.fit_transform(self.ai_df[self.feature_cols])
        human_pca = pca.transform(self.human_df[self.feature_cols])
        
        try:
            from scipy.spatial import ConvexHull
            ai_hull = ConvexHull(ai_pca)
            human_hull = ConvexHull(human_pca)
            
            results['ai_hull_volume'] = ai_hull.volume
            results['human_hull_volume'] = human_hull.volume
            results['hull_volume_ratio'] = ai_hull.volume / human_hull.volume
            
            print(f"\nConvex hull volume ratio (AI/Human): {results['hull_volume_ratio']:.3f}")
        except Exception as e:
            print(f"\nConvex hull computation failed: {e}")
            results['hull_volume_ratio'] = np.nan
        
        # 3. Effective dimensionality (participation ratio)
        ai_cov = np.cov(self.ai_df[self.feature_cols].values.T)
        human_cov = np.cov(self.human_df[self.feature_cols].values.T)
        
        ai_eigvals = np.linalg.eigvals(ai_cov)
        human_eigvals = np.linalg.eigvals(human_cov)
        
        ai_pr = (np.sum(ai_eigvals)**2) / np.sum(ai_eigvals**2)
        human_pr = (np.sum(human_eigvals)**2) / np.sum(human_eigvals**2)
        
        results['ai_participation_ratio'] = ai_pr
        results['human_participation_ratio'] = human_pr
        
        print(f"\nEffective dimensionality (participation ratio):")
        print(f"  AI: {ai_pr:.2f}")
        print(f"  Human: {human_pr:.2f}")
        
        if ai_pr < human_pr:
            print("  ✓ AI tracks use fewer effective dimensions")
        
        self.results['coverage'] = results
        return results
    
    def analyze_entropy_diversity(self):
        """Test H4: AI tracks have lower distributional entropy"""
        print("\n" + "="*60)
        print("ENTROPY & DIVERSITY ANALYSIS")
        print("="*60)
        
        results = {}
        
        # For each feature, compute histogram entropy
        entropy_ratios = []
        
        for feature in self.feature_cols:
            # Discretize into bins
            bins = 20
            ai_hist, _ = np.histogram(self.ai_df[feature].dropna(), bins=bins, density=True)
            human_hist, _ = np.histogram(self.human_df[feature].dropna(), bins=bins, density=True)
            
            # Compute entropy
            ai_hist = ai_hist + 1e-10  # Avoid log(0)
            human_hist = human_hist + 1e-10
            
            ai_entropy = -np.sum(ai_hist * np.log2(ai_hist))
            human_entropy = -np.sum(human_hist * np.log2(human_hist))
            
            entropy_ratio = ai_entropy / human_entropy if human_entropy > 0 else np.nan
            entropy_ratios.append(entropy_ratio)
        
        results['mean_entropy_ratio'] = np.nanmean(entropy_ratios)
        results['median_entropy_ratio'] = np.nanmedian(entropy_ratios)
        
        print(f"\nMean entropy ratio (AI/Human): {results['mean_entropy_ratio']:.3f}")
        print(f"Features where AI has lower entropy: {sum(r < 1 for r in entropy_ratios if not np.isnan(r))} / {len(entropy_ratios)}")
        
        self.results['entropy'] = results
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive summary of all analyses"""
        print("\n" + "="*80)
        print("COMPREHENSIVE HOMOGENIZATION REPORT")
        print("="*80)
        
        print("\n### HYPOTHESIS TESTS ###")
        
        # H1: Variance
        var_df = self.results['variance']
        pct_lower_var = (var_df['variance_ratio'] < 1.0).mean() * 100
        print(f"\nH1 (Lower Variance): {pct_lower_var:.1f}% of features show AI < Human variance")
        
        # H2: Pairwise distances
        dist_results = self.results['distances']
        print(f"\nH2 (Tighter Clustering):")
        print(f"  AI mean distance: {dist_results['ai_mean_distance']:.3f}")
        print(f"  Human mean distance: {dist_results['human_mean_distance']:.3f}")
        print(f"  Cohen's d: {dist_results['cohens_d']:.3f}")
        print(f"  p-value: {dist_results['mannwhitney_pvalue']:.4f}")
        
        # H3: Coverage
        cov_results = self.results['coverage']
        print(f"\nH3 (Feature Space Coverage):")
        print(f"  Range ratio: {cov_results['mean_range_ratio']:.3f}")
        if 'hull_volume_ratio' in cov_results:
            print(f"  Hull volume ratio: {cov_results['hull_volume_ratio']:.3f}")
        print(f"  Participation ratio (AI): {cov_results['ai_participation_ratio']:.2f}")
        print(f"  Participation ratio (Human): {cov_results['human_participation_ratio']:.2f}")
        
        # H4: Entropy
        ent_results = self.results['entropy']
        print(f"\nH4 (Distributional Diversity):")
        print(f"  Entropy ratio: {ent_results['mean_entropy_ratio']:.3f}")
        
        # Overall verdict
        print("\n" + "="*80)
        print("VERDICT:")
        
        homogenization_score = 0
        if pct_lower_var > 60:
            homogenization_score += 1
        if dist_results['cohens_d'] < -0.2:
            homogenization_score += 1
        if cov_results['mean_range_ratio'] < 0.9:
            homogenization_score += 1
        if ent_results['mean_entropy_ratio'] < 0.95:
            homogenization_score += 1
        
        if homogenization_score >= 3:
            print("✓✓✓ STRONG EVIDENCE of AI music homogenization")
        elif homogenization_score == 2:
            print("✓✓ MODERATE EVIDENCE of AI music homogenization")
        elif homogenization_score == 1:
            print("✓ WEAK EVIDENCE of AI music homogenization")
        else:
            print("✗ NO EVIDENCE of AI music homogenization")
        
        print(f"Score: {homogenization_score}/4 criteria met")
        print("="*80)
    
    def save_results(self, output_prefix='homogenization'):
        """Save all results to CSV files"""
        # Variance analysis
        self.results['variance'].to_csv(f'{output_prefix}_variance.csv', index=False)
        
        # Distance analysis
        pd.DataFrame([self.results['distances']]).to_csv(f'{output_prefix}_distances.csv', index=False)
        
        # Coverage analysis
        pd.DataFrame([self.results['coverage']]).to_csv(f'{output_prefix}_coverage.csv', index=False)
        
        # Entropy analysis
        pd.DataFrame([self.results['entropy']]).to_csv(f'{output_prefix}_entropy.csv', index=False)
        
        print(f"\n✓ Results saved with prefix '{output_prefix}_'")


if __name__ == "__main__":
    # new file path
    project_root = Path(__file__).resolve().parent.parent
    features_csv = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"
    
    print("Loading feature data...")
    df = pd.read_csv(features_csv)
    
    print(f"Loaded {len(df)} tracks")
    print(f"  AI: {len(df[df['side']=='AI'])}")
    print(f"  Human: {len(df[df['side']=='Human'])}")
    
    # Run analysis
    analyzer = HomogenizationAnalyzer(df)
    analyzer.analyze_variance_homogeneity()
    analyzer.analyze_pairwise_distances()
    analyzer.analyze_feature_space_coverage()
    analyzer.analyze_entropy_diversity()
    analyzer.generate_summary_report()
    analyzer.save_results()