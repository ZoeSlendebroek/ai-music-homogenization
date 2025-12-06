"""
Validation of AI-Human Matched Pairs
Tests whether matched pairs are truly similar and validates matching quality
"""

import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PairValidator:
    """Validate quality of AI-human matched pairs"""
    
    def __init__(self, features_csv):
        self.features = pd.read_csv(features_csv)
        
        # UPDATED: Use 'side' column instead of 'label'
        self.ai_features = self.features[self.features['side'] == 'AI'].copy()
        self.human_features = self.features[self.features['side'] == 'Human'].copy()
        
        # UPDATED: Exclude new metadata columns
        self.feature_cols = [col for col in self.features.columns 
                            if col not in ['filename', 'side', 'pair_key', 'track_id']]
        
        # Standardize features
        self.scaler = StandardScaler()
        all_features = self.features[self.feature_cols].values
        self.scaler.fit(all_features)
        
        print(f"Loaded {len(self.ai_features)} AI tracks and {len(self.human_features)} Human tracks")
        print(f"Using {len(self.feature_cols)} features for validation")
    
    def validate_within_pair_similarity(self):
        """Test that matched pairs are actually similar"""
        print("\n" + "="*60)
        print("WITHIN-PAIR SIMILARITY VALIDATION")
        print("="*60)
        
        pair_distances = []
        
        # UPDATED: Group by pair_key to match AI-Human pairs
        unique_pairs = self.ai_features['pair_key'].unique()
        
        for pair_key in unique_pairs:
            # Get AI track for this pair
            ai_track = self.ai_features[self.ai_features['pair_key'] == pair_key]
            
            if len(ai_track) == 0:
                print(f"Warning: AI track not found for pair: {pair_key}")
                continue
            
            # Get matched human track
            human_track = self.human_features[self.human_features['pair_key'] == pair_key]
            
            if len(human_track) == 0:
                print(f"Warning: Human track not found for pair: {pair_key}")
                continue
            
            # Extract feature vectors
            ai_vec = ai_track[self.feature_cols].values[0]
            human_vec = human_track[self.feature_cols].values[0]
            
            # Standardize
            ai_vec_std = self.scaler.transform([ai_vec])[0]
            human_vec_std = self.scaler.transform([human_vec])[0]
            
            # Compute distance
            dist = euclidean(ai_vec_std, human_vec_std)
            
            # UPDATED: Get tempo difference if available
            tempo_diff = abs(ai_track['tempo'].values[0] - human_track['tempo'].values[0]) if 'tempo' in self.feature_cols else np.nan
            
            pair_distances.append({
                'pair_key': pair_key,
                'ai_filename': ai_track['filename'].values[0],
                'human_filename': human_track['filename'].values[0],
                'distance': dist,
                'tempo_diff': tempo_diff
            })
        
        pair_df = pd.DataFrame(pair_distances)
        
        print(f"\nSuccessfully validated {len(pair_df)} pairs")
        print(f"Mean within-pair distance: {pair_df['distance'].mean():.3f} ± {pair_df['distance'].std():.3f}")
        print(f"Median within-pair distance: {pair_df['distance'].median():.3f}")
        print(f"Distance range: [{pair_df['distance'].min():.3f}, {pair_df['distance'].max():.3f}]")
        
        return pair_df
    
    def compare_to_random_pairs(self, pair_df, n_permutations=1000):
        """Compare matched pairs to random pairings (permutation test)"""
        print("\n" + "="*60)
        print("PERMUTATION TEST: MATCHED vs RANDOM PAIRS")
        print("="*60)
        
        matched_distances = pair_df['distance'].values
        matched_mean = np.mean(matched_distances)
        
        # Generate random pairings
        random_means = []
        
        for _ in range(n_permutations):
            # Randomly shuffle human tracks
            shuffled_human = self.human_features.sample(frac=1).reset_index(drop=True)
            
            random_distances = []
            for i in range(min(len(self.ai_features), len(shuffled_human))):
                ai_vec = self.ai_features.iloc[i][self.feature_cols].values
                human_vec = shuffled_human.iloc[i][self.feature_cols].values
                
                ai_vec_std = self.scaler.transform([ai_vec])[0]
                human_vec_std = self.scaler.transform([human_vec])[0]
                
                dist = euclidean(ai_vec_std, human_vec_std)
                random_distances.append(dist)
            
            random_means.append(np.mean(random_distances))
        
        random_means = np.array(random_means)
        
        # Compute p-value: how many random pairings have lower mean distance?
        p_value = np.sum(random_means <= matched_mean) / n_permutations
        
        # Effect size
        effect_size = (np.mean(random_means) - matched_mean) / np.std(random_means)
        
        print(f"\nMatched pairs mean distance: {matched_mean:.3f}")
        print(f"Random pairs mean distance: {np.mean(random_means):.3f} ± {np.std(random_means):.3f}")
        print(f"Difference: {np.mean(random_means) - matched_mean:.3f}")
        print(f"Effect size (standardized): {effect_size:.3f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n✓ Matched pairs are significantly more similar than random pairs")
        else:
            print("\n✗ WARNING: Matched pairs not significantly better than random")
        
        return {
            'matched_mean': matched_mean,
            'random_mean': np.mean(random_means),
            'random_std': np.std(random_means),
            'effect_size': effect_size,
            'p_value': p_value
        }
    
    def analyze_pair_quality_distribution(self, pair_df):
        """Analyze distribution of pair quality"""
        print("\n" + "="*60)
        print("PAIR QUALITY DISTRIBUTION")
        print("="*60)
        
        # Identify outlier pairs (poor matches)
        q75 = pair_df['distance'].quantile(0.75)
        q25 = pair_df['distance'].quantile(0.25)
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        
        outliers = pair_df[pair_df['distance'] > outlier_threshold]
        
        print(f"\nPairs exceeding outlier threshold ({outlier_threshold:.3f}):")
        print(f"  Count: {len(outliers)}")
        
        if len(outliers) > 0:
            print("\nPoorest matches:")
            print(outliers.sort_values('distance', ascending=False)[
                ['ai_filename', 'human_filename', 'distance', 'tempo_diff']
            ].head())
        
        # Quality tiers
        print("\nPair quality distribution:")
        print(f"  Excellent (distance < 5): {(pair_df['distance'] < 5).sum()} pairs")
        print(f"  Good (5-7): {((pair_df['distance'] >= 5) & (pair_df['distance'] < 7)).sum()} pairs")
        print(f"  Fair (7-9): {((pair_df['distance'] >= 7) & (pair_df['distance'] < 9)).sum()} pairs")
        print(f"  Poor (9+): {(pair_df['distance'] >= 9).sum()} pairs")
        
        return outliers
    
    def validate_matching_constraints(self, pair_df):
        """Validate that matching constraints were satisfied"""
        print("\n" + "="*60)
        print("MATCHING CONSTRAINTS VALIDATION")
        print("="*60)
        
        # Check tempo constraint
        tempo_violations = pair_df[pair_df['tempo_diff'] > 8]
        print(f"\nTempo constraint (≤8 BPM):")
        print(f"  Violations: {len(tempo_violations)} / {len(pair_df)}")
        
        if len(tempo_violations) > 0:
            print("  WARNING: Tempo constraints violated for:")
            print(tempo_violations[['ai_filename', 'human_filename', 'tempo_diff']])
        else:
            print("  ✓ All pairs satisfy tempo constraint")
        
        # Check for duplicate human tracks
        human_counts = pair_df['human_filename'].value_counts()
        duplicates = human_counts[human_counts > 1]
        
        print(f"\nUnique human tracks constraint:")
        print(f"  Duplicate uses: {len(duplicates)}")
        
        if len(duplicates) > 0:
            print("  WARNING: Some human tracks used multiple times:")
            print(duplicates)
        else:
            print("  ✓ All human tracks used at most once")
        
        return {
            'tempo_violations': len(tempo_violations),
            'duplicate_humans': len(duplicates)
        }
    
    def compute_pair_level_statistics(self, pair_df):
        """Compute statistics for each pair for downstream mixed-effects models"""
        print("\n" + "="*60)
        print("PAIR-LEVEL STATISTICS")
        print("="*60)
        
        # For each pair, compute feature-wise differences
        pair_stats = []
        
        for _, row in pair_df.iterrows():
            pair_key = row['pair_key']
            
            ai_track = self.ai_features[self.ai_features['pair_key'] == pair_key]
            human_track = self.human_features[self.human_features['pair_key'] == pair_key]
            
            if len(ai_track) == 0 or len(human_track) == 0:
                continue
            
            # Compute differences for key features
            stats_dict = {
                'pair_key': pair_key,
                'ai_filename': ai_track['filename'].values[0],
                'human_filename': human_track['filename'].values[0]
            }
            
            for feature in ['tempo', 'spectral_centroid_mean', 'rms_mean', 
                          'chroma_stft_mean', 'onset_density']:
                if feature in self.feature_cols:
                    ai_val = ai_track[feature].values[0]
                    human_val = human_track[feature].values[0]
                    stats_dict[f'{feature}_diff'] = abs(ai_val - human_val)
                    stats_dict[f'{feature}_ai'] = ai_val
                    stats_dict[f'{feature}_human'] = human_val
            
            pair_stats.append(stats_dict)
        
        pair_stats_df = pd.DataFrame(pair_stats)
        
        print(f"\nComputed statistics for {len(pair_stats_df)} pairs")
        print("\nMean absolute differences (key features):")
        
        for col in pair_stats_df.columns:
            if col.endswith('_diff'):
                print(f"  {col}: {pair_stats_df[col].mean():.3f}")
        
        return pair_stats_df
    
    def generate_validation_report(self, pair_df, permutation_results):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("PAIR MATCHING VALIDATION REPORT")
        print("="*80)
        
        print("\n### MATCHING QUALITY ###")
        print(f"Total pairs: {len(pair_df)}")
        print(f"Mean within-pair distance: {pair_df['distance'].mean():.3f}")
        print(f"Std within-pair distance: {pair_df['distance'].std():.3f}")
        
        print("\n### COMPARISON TO RANDOM ###")
        print(f"Matched mean: {permutation_results['matched_mean']:.3f}")
        print(f"Random mean: {permutation_results['random_mean']:.3f}")
        print(f"Improvement: {permutation_results['random_mean'] - permutation_results['matched_mean']:.3f}")
        print(f"Effect size: {permutation_results['effect_size']:.3f}")
        print(f"p-value: {permutation_results['p_value']:.4f}")
        
        if permutation_results['p_value'] < 0.05:
            print("\n✓✓ VALID MATCHING: Pairs significantly better than chance")
        else:
            print("\n✗✗ INVALID MATCHING: Pairs not better than random")
        
        print("="*80)


if __name__ == "__main__":
    # UPDATED: Use new file path
    project_root = Path(__file__).resolve().parent.parent
    features_csv = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"
    
    # Load data
    validator = PairValidator(features_csv)
    
    # Run validation
    pair_df = validator.validate_within_pair_similarity()
    permutation_results = validator.compare_to_random_pairs(pair_df, n_permutations=1000)
    validator.analyze_pair_quality_distribution(pair_df)
    validator.validate_matching_constraints(pair_df)
    pair_stats = validator.compute_pair_level_statistics(pair_df)
    
    # Generate report
    validator.generate_validation_report(pair_df, permutation_results)
    
    # Save results
    pair_df.to_csv('pair_validation_results.csv', index=False)
    pair_stats.to_csv('pair_statistics.csv', index=False)
    
    print("\n✓ Validation complete. Results saved.")