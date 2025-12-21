"""
Classification and Feature Importance Analysis
Train classifiers to discriminate AI from human tracks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AIHumanClassifier:
    """Train classifiers to discriminate AI from human tracks"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        
        # UPDATED: Exclude new metadata columns
        self.feature_cols = [col for col in features_df.columns 
                            if col not in ['filename', 'side', 'pair_key', 'track_id']]
        
        # Prepare data
        self.X = self.df[self.feature_cols].values
        self.y = (self.df['side'] == 'AI').astype(int).values
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Dataset: {len(self.X)} tracks, {len(self.feature_cols)} features")
        print(f"  AI: {np.sum(self.y)} tracks")
        print(f"  Human: {len(self.y) - np.sum(self.y)} tracks")
        
        self.results = {}
    
    def evaluate_classifiers(self, cv_folds=5):
        """Evaluate multiple classifier types with cross-validation"""
        print("\n" + "="*60)
        print("CLASSIFIER EVALUATION (Cross-Validation)")
        print("="*60)
        
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = []
        
        for name, clf in classifiers.items():
            print(f"\nEvaluating {name}...")
            
            # Cross-validation
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            scores = cross_validate(clf, self.X_scaled, self.y, cv=cv, scoring=scoring)
            
            result = {
                'classifier': name,
                'accuracy_mean': scores['test_accuracy'].mean(),
                'accuracy_std': scores['test_accuracy'].std(),
                'precision_mean': scores['test_precision'].mean(),
                'precision_std': scores['test_precision'].std(),
                'recall_mean': scores['test_recall'].mean(),
                'recall_std': scores['test_recall'].std(),
                'f1_mean': scores['test_f1'].mean(),
                'f1_std': scores['test_f1'].std(),
                'auc_mean': scores['test_roc_auc'].mean(),
                'auc_std': scores['test_roc_auc'].std()
            }
            
            results.append(result)
            
            print(f"  Accuracy: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
            print(f"  AUC: {result['auc_mean']:.3f} ± {result['auc_std']:.3f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('auc_mean', ascending=False)
        
        print("\n" + "="*60)
        print("CLASSIFIER RANKING (by AUC)")
        print("="*60)
        print(results_df[['classifier', 'accuracy_mean', 'auc_mean']])
        
        self.results['classifier_comparison'] = results_df
        return results_df
    
    def train_best_model_and_extract_importance(self):
        """Train best model on full data and extract feature importance"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Train Random Forest (best for feature importance)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        rf.fit(self.X_scaled, self.y)
        
        # Extract feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'feature': [self.feature_cols[i] for i in indices],
            'importance': importances[indices],
            'rank': range(1, len(indices) + 1)
        })
        
        print("\nTop 20 Most Important Features for AI Detection:")
        print(importance_df.head(20))
        
        # Cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        n_features_90pct = (importance_df['cumulative_importance'] <= 0.90).sum()
        print(f"\nFeatures needed for 90% importance: {n_features_90pct} / {len(self.feature_cols)}")
        
        self.results['feature_importance'] = importance_df
        self.results['trained_model'] = rf
        
        return importance_df
    
    def analyze_distinguishability(self):
        """Analyze how distinguishable AI vs human tracks are"""
        print("\n" + "="*60)
        print("DISTINGUISHABILITY ANALYSIS")
        print("="*60)
        
        # Train model with cross-validation predictions
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        y_pred_proba = np.zeros(len(self.y))
        
        for train_idx, test_idx in cv.split(self.X_scaled, self.y):
            rf.fit(self.X_scaled[train_idx], self.y[train_idx])
            y_pred_proba[test_idx] = rf.predict_proba(self.X_scaled[test_idx])[:, 1]
        
        # Compute AUC
        auc = roc_auc_score(self.y, y_pred_proba)
        
        # Compute confidence for each track
        confidence = np.abs(y_pred_proba - 0.5) * 2  # Convert to 0-1 scale
        
        # Split by actual label
        ai_confidence = confidence[self.y == 1]
        human_confidence = confidence[self.y == 0]
        
        print(f"\nOverall AUC: {auc:.3f}")
        print(f"\nMean prediction confidence:")
        print(f"  AI tracks: {np.mean(ai_confidence):.3f} ± {np.std(ai_confidence):.3f}")
        print(f"  Human tracks: {np.mean(human_confidence):.3f} ± {np.std(human_confidence):.3f}")
        
        # use 'side' column
        track_results = pd.DataFrame({
            'filename': self.df['filename'],
            'side': self.df['side'],
            'predicted_prob_ai': y_pred_proba,
            'confidence': confidence,
            'correct': (y_pred_proba >= 0.5) == self.y
        })
        
        print("\nMost AI-like tracks:")
        print(track_results.nlargest(5, 'predicted_prob_ai')[['filename', 'side', 'predicted_prob_ai']])
        
        print("\nMost human-like tracks:")
        print(track_results.nsmallest(5, 'predicted_prob_ai')[['filename', 'side', 'predicted_prob_ai']])
        
        print("\nLeast distinguishable tracks (closest to 0.5):")
        track_results['distance_from_boundary'] = np.abs(track_results['predicted_prob_ai'] - 0.5)
        print(track_results.nsmallest(5, 'distance_from_boundary')[['filename', 'side', 'predicted_prob_ai']])
        
        self.results['distinguishability'] = track_results
        
        # Interpretation
        if auc > 0.90:
            print("\n✓✓✓ HIGHLY DISTINGUISHABLE: AI tracks are very different from human")
        elif auc > 0.75:
            print("\n✓✓ MODERATELY DISTINGUISHABLE: AI tracks show detectable patterns")
        elif auc > 0.60:
            print("\n✓ WEAKLY DISTINGUISHABLE: Subtle differences exist")
        else:
            print("\n✗ NOT DISTINGUISHABLE: AI and human tracks are acoustically similar")
        
        return track_results
    
    def analyze_feature_categories(self):
        """Group features by category and analyze which categories matter most"""
        print("\n" + "="*60)
        print("FEATURE CATEGORY ANALYSIS")
        print("="*60)
        
        importance_df = self.results['feature_importance']
        
        # Feature categories
        def categorize_feature(feature_name):
            if 'mfcc' in feature_name:
                return 'Timbral (MFCC)'
            elif any(x in feature_name for x in ['spectral', 'zcr', 'harmonic', 'percussive']):
                return 'Spectral/Timbral'
            elif any(x in feature_name for x in ['tempo', 'onset', 'ioi', 'beat']):
                return 'Rhythmic'
            elif any(x in feature_name for x in ['chroma', 'tonnetz']):
                return 'Harmonic'
            elif any(x in feature_name for x in ['rms', 'dynamic', 'crest']):
                return 'Dynamic'
            elif any(x in feature_name for x in ['repetition', 'self_similarity']):
                return 'Structural'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # Aggregate by category
        category_importance = importance_df.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
        category_importance = category_importance.sort_values('sum', ascending=False)
        
        print("\nFeature Category Importance:")
        print(category_importance)
        
        # Top feature per category
        print("\nTop feature per category:")
        for category in category_importance.index:
            top_feature = importance_df[importance_df['category'] == category].iloc[0]
            print(f"  {category}: {top_feature['feature']} (importance: {top_feature['importance']:.4f})")
        
        self.results['category_importance'] = category_importance
        return category_importance
    
    def test_paired_distinguishability(self):
        """Test if matched pairs are distinguishable"""
        print("\n" + "="*60)
        print("PAIRED DISTINGUISHABILITY ANALYSIS")
        print("="*60)
        
        track_results = self.results['distinguishability']
        
        # Group by pair_key
        unique_pairs = self.df['pair_key'].unique()
        
        pair_predictions = []
        
        for pair_key in unique_pairs:
            # predictions for this pair
            ai_pred = track_results[(self.df['pair_key'] == pair_key) & (self.df['side'] == 'AI')]
            human_pred = track_results[(self.df['pair_key'] == pair_key) & (self.df['side'] == 'Human')]
            
            if len(ai_pred) == 0 or len(human_pred) == 0:
                continue
            
            pair_predictions.append({
                'pair_key': pair_key,
                'ai_filename': ai_pred['filename'].values[0],
                'human_filename': human_pred['filename'].values[0],
                'ai_prob': ai_pred['predicted_prob_ai'].values[0],
                'human_prob': human_pred['predicted_prob_ai'].values[0],
                'pair_separation': ai_pred['predicted_prob_ai'].values[0] - 
                                  human_pred['predicted_prob_ai'].values[0]
            })
        
        pair_df = pd.DataFrame(pair_predictions)
        
        print(f"\nAnalyzed {len(pair_df)} matched pairs")
        print(f"Mean AI probability: {pair_df['ai_prob'].mean():.3f}")
        print(f"Mean Human probability: {pair_df['human_prob'].mean():.3f}")
        print(f"Mean pair separation: {pair_df['pair_separation'].mean():.3f}")
        
        # Test if separation is significant
        t_stat, p_value = stats.ttest_1samp(pair_df['pair_separation'], 0)
        
        print(f"\nOne-sample t-test (H0: separation = 0):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  ✓ Matched pairs are distinguishable despite acoustic similarity")
        else:
            print("  ✗ Matched pairs are not reliably distinguishable")
        
        self.results['pair_distinguishability'] = pair_df
        return pair_df
    
    def generate_classification_report(self):
        """Generate comprehensive classification report"""
        print("\n" + "="*80)
        print("CLASSIFICATION ANALYSIS REPORT")
        print("="*80)
        
        # Best classifier
        clf_comparison = self.results['classifier_comparison']
        best_clf = clf_comparison.iloc[0]
        
        print(f"\n### BEST CLASSIFIER ###")
        print(f"Model: {best_clf['classifier']}")
        print(f"Accuracy: {best_clf['accuracy_mean']:.3f} ± {best_clf['accuracy_std']:.3f}")
        print(f"AUC: {best_clf['auc_mean']:.3f} ± {best_clf['auc_std']:.3f}")
        
        # Feature importance
        importance_df = self.results['feature_importance']
        print(f"\n### TOP 5 DISCRIMINATIVE FEATURES ###")
        for idx, row in importance_df.head(5).iterrows():
            print(f"{row['rank']}. {row['feature']}: {row['importance']:.4f}")
        
        # Category analysis
        category_imp = self.results['category_importance']
        print(f"\n### MOST IMPORTANT FEATURE CATEGORIES ###")
        print(category_imp['sum'].head())
        
        print("\n" + "="*80)
    
    def save_results(self, output_prefix='classification'):
        """Save all classification results"""
        self.results['classifier_comparison'].to_csv(f'{output_prefix}_comparison.csv', index=False)
        self.results['feature_importance'].to_csv(f'{output_prefix}_importance.csv', index=False)
        self.results['distinguishability'].to_csv(f'{output_prefix}_tracks.csv', index=False)
        
        if 'pair_distinguishability' in self.results:
            self.results['pair_distinguishability'].to_csv(f'{output_prefix}_pairs.csv', index=False)
        
        print(f"\n✓ Classification results saved with prefix '{output_prefix}_'")


if __name__ == "__main__":
    # UPDATED: new file path
    project_root = Path(__file__).resolve().parent.parent
    features_csv = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"
    
    # Load features
    print("Loading feature data...")
    df = pd.read_csv(features_csv)
    
    # Initialize classifier
    classifier = AIHumanClassifier(df)
    
    # Run analyses
    classifier.evaluate_classifiers(cv_folds=5)
    classifier.train_best_model_and_extract_importance()
    classifier.analyze_distinguishability()
    classifier.analyze_feature_categories()
    
    # Test paired distinguishability
    try:
        classifier.test_paired_distinguishability()
    except Exception as e:
        print(f"\nSkipping paired analysis: {e}")
    
    # Generate report
    classifier.generate_classification_report()
    
    # Save results
    classifier.save_results()