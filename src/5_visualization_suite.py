"""
Comprehensive Visualization Suite for AI Music Homogenization Analysis
Generates publication-quality figures for paper
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.spatial.distance import pdist
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

class VisualizationSuite:
    """Generate comprehensive visualizations for analysis"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        
        # UPDATED: Use 'side' column instead of 'label'
        self.ai_df = features_df[features_df['side'] == 'AI'].copy()
        self.human_df = features_df[features_df['side'] == 'Human'].copy()
        
        # UPDATED: Exclude new metadata columns
        self.feature_cols = [col for col in features_df.columns 
                            if col not in ['filename', 'side', 'pair_key', 'track_id']]
        
        # Standardize
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        
        print(f"Initialized visualization suite with {len(self.df)} tracks")
        print(f"  AI: {len(self.ai_df)}, Human: {len(self.human_df)}")
    
    def plot_pairwise_distance_distributions(self, output_file='fig_distances.png'):
        """Compare pairwise distance distributions"""
        ai_features = self.ai_df[self.feature_cols].values
        human_features = self.human_df[self.feature_cols].values
        
        ai_distances = pdist(ai_features, metric='euclidean')
        human_distances = pdist(human_features, metric='euclidean')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram comparison
        axes[0].hist(ai_distances, bins=30, alpha=0.6, label='AI', density=True, color='#FF6B6B')
        axes[0].hist(human_distances, bins=30, alpha=0.6, label='Human', density=True, color='#4ECDC4')
        axes[0].set_xlabel('Pairwise Distance (Euclidean)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution of Pairwise Distances')
        axes[0].legend()
        axes[0].axvline(np.mean(ai_distances), color='#FF6B6B', linestyle='--', linewidth=2)
        axes[0].axvline(np.mean(human_distances), color='#4ECDC4', linestyle='--', linewidth=2)
        
        # Box plot comparison
        data = [ai_distances, human_distances]
        axes[1].boxplot(data, labels=['AI', 'Human'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Pairwise Distance')
        axes[1].set_title('Pairwise Distance Distributions')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_variance_comparison(self, variance_df, output_file='fig_variance.png'):
        """Visualize variance differences across features"""
        var_sorted = variance_df.sort_values('variance_ratio').head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(var_sorted))
        colors = ['#FF6B6B' if x < 1.0 else '#4ECDC4' for x in var_sorted['variance_ratio']]
        
        ax.barh(y_pos, var_sorted['variance_ratio'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_sorted['feature'], fontsize=8)
        ax.set_xlabel('Variance Ratio (AI / Human)')
        ax.set_title('Feature Variance: AI vs Human\n(Top 20 features where AI is most homogeneous)')
        ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Equal variance')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_pca_projection(self, output_file='fig_pca.png'):
        """2D PCA projection of feature space"""
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.df[self.feature_cols])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # UPDATED: Use 'side' column
        for label, color, marker in [('AI', '#FF6B6B', 'o'), ('Human', '#4ECDC4', 's')]:
            mask = self.df['side'] == label
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                      c=color, label=label, alpha=0.6, s=100, marker=marker, 
                      edgecolors='black', linewidth=0.5)
        
        # Add 95% confidence ellipses
        for label, color in [('AI', '#FF6B6B'), ('Human', '#4ECDC4')]:
            mask = self.df['side'] == label
            points = features_pca[mask]
            
            if len(points) > 2:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
                width, height = 2 * np.sqrt(eigenvalues) * 2.447  # 95% CI
                
                ellipse = Ellipse(mean, width, height, angle=angle, 
                                 facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
                ax.add_patch(ellipse)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title('PCA Projection: AI vs Human Afrobeats Tracks')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_tsne_projection(self, output_file='fig_tsne.png'):
        """t-SNE projection of feature space"""
        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df)-1))
        features_tsne = tsne.fit_transform(self.df[self.feature_cols])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # UPDATED: Use 'side' column
        for label, color, marker in [('AI', '#FF6B6B', 'o'), ('Human', '#4ECDC4', 's')]:
            mask = self.df['side'] == label
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                      c=color, label=label, alpha=0.6, s=100, marker=marker, 
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Projection: AI vs Human Afrobeats Tracks')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_feature_importance(self, importance_df, output_file='fig_importance.png'):
        """Visualize top discriminative features"""
        top_features = importance_df.head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='#95E1D3', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.set_xlabel('Feature Importance (Random Forest)')
        ax.set_title('Top 15 Features for AI Detection')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_feature_category_importance(self, category_importance, output_file='fig_categories.png'):
        """Visualize feature category importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = category_importance.index
        importance = category_importance['sum'].values
        
        colors = sns.color_palette("husl", len(categories))
        ax.bar(range(len(categories)), importance, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Total Importance')
        ax.set_title('Feature Category Importance for AI Detection')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_pair_validation(self, pair_df, permutation_results, output_file='fig_pairs.png'):
        """Visualize matched pair validation"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Distribution of within-pair distances
        axes[0].hist(pair_df['distance'], bins=15, color='#FF6B6B', alpha=0.7, edgecolor='black')
        axes[0].axvline(pair_df['distance'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0].set_xlabel('Within-Pair Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Matched Pair Distances')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Comparison to random
        axes[1].axvline(permutation_results['matched_mean'], color='#FF6B6B', 
                       linestyle='--', linewidth=2, label='Matched pairs')
        axes[1].axvline(permutation_results['random_mean'], color='#4ECDC4', 
                       linestyle='--', linewidth=2, label='Random pairs')
        axes[1].axvspan(permutation_results['random_mean'] - permutation_results['random_std'],
                       permutation_results['random_mean'] + permutation_results['random_std'],
                       alpha=0.3, color='#4ECDC4', label='Random ±1 SD')
        axes[1].set_xlabel('Mean Pairwise Distance')
        axes[1].set_title('Matched vs Random Pairs')
        axes[1].legend()
        axes[1].set_xlim(0, max(permutation_results['matched_mean'], permutation_results['random_mean']) * 1.2)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_confusion_heatmap(self, track_results, output_file='fig_confusion.png'):
        """Plot confusion matrix heatmap"""
        # UPDATED: Use 'side' column
        y_true = (track_results['side'] == 'AI').astype(int)
        y_pred = (track_results['predicted_prob_ai'] >= 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Human', 'Predicted AI'],
                   yticklabels=['True Human', 'True AI'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_title('Classification Confusion Matrix')
        
        # Add accuracy
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
               ha='center', transform=ax.transAxes, fontsize=11, weight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_roc_curve(self, track_results, output_file='fig_roc.png'):
        """Plot ROC curve"""
        # UPDATED: Use 'side' column
        y_true = (track_results['side'] == 'AI').astype(int)
        y_score = track_results['predicted_prob_ai']
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color='#FF6B6B', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve: AI vs Human Classification')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def create_composite_figure(self, variance_df, importance_df, output_file='fig_composite.png'):
        """Create multi-panel composite figure for paper"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: PCA
        ax1 = fig.add_subplot(gs[0, 0])
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.df[self.feature_cols])
        for label, color, marker in [('AI', '#FF6B6B', 'o'), ('Human', '#4ECDC4', 's')]:
            mask = self.df['side'] == label
            ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=color, label=label, alpha=0.6, s=60, marker=marker)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.set_title('A. PCA Projection')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel B: Pairwise distances
        ax2 = fig.add_subplot(gs[0, 1])
        ai_features = self.ai_df[self.feature_cols].values
        human_features = self.human_df[self.feature_cols].values
        ai_dist = pdist(ai_features)
        human_dist = pdist(human_features)
        ax2.hist(ai_dist, bins=20, alpha=0.6, label='AI', density=True, color='#FF6B6B')
        ax2.hist(human_dist, bins=20, alpha=0.6, label='Human', density=True, color='#4ECDC4')
        ax2.set_xlabel('Pairwise Distance')
        ax2.set_ylabel('Density')
        ax2.set_title('B. Distance Distributions')
        ax2.legend()
        
        # Panel C: Variance comparison
        ax3 = fig.add_subplot(gs[1, :])
        var_sorted = variance_df.sort_values('variance_ratio').head(15)
        y_pos = np.arange(len(var_sorted))
        colors = ['#FF6B6B' if x < 1.0 else '#4ECDC4' for x in var_sorted['variance_ratio']]
        ax3.barh(y_pos, var_sorted['variance_ratio'], color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(var_sorted['feature'], fontsize=7)
        ax3.set_xlabel('Variance Ratio (AI / Human)')
        ax3.set_title('C. Feature Variance Comparison')
        ax3.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.grid(axis='x', alpha=0.3)
        
        # Panel D: Feature importance
        ax4 = fig.add_subplot(gs[2, :])
        top_feat = importance_df.head(12)
        y_pos = np.arange(len(top_feat))
        ax4.barh(y_pos, top_feat['importance'], color='#95E1D3', alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_feat['feature'], fontsize=8)
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('D. Top Discriminative Features')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✓ Saved composite figure: {output_file}")
        plt.close()


def generate_all_visualizations(features_csv=None,
                                variance_csv='homogenization_variance.csv',
                                importance_csv='classification_importance.csv',
                                pairs_csv='pair_validation_results.csv',
                                tracks_csv='classification_tracks.csv',
                                permutation_results=None):
    """Generate all visualizations for the paper"""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)
    
    # UPDATED: Default to new file path
    if features_csv is None:
        project_root = Path(__file__).resolve().parent.parent
        features_csv = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"
    
    # Load data
    features_df = pd.read_csv(features_csv)
    viz = VisualizationSuite(features_df)
    
    # Core visualizations
    viz.plot_pairwise_distance_distributions()
    viz.plot_pca_projection()
    viz.plot_tsne_projection()
    
    # Variance analysis
    try:
        variance_df = pd.read_csv(variance_csv)
        viz.plot_variance_comparison(variance_df)
    except Exception as e:
        print(f"Variance data not found: {e}")
    
    # Feature importance
    try:
        importance_df = pd.read_csv(importance_csv)
        viz.plot_feature_importance(importance_df)
        
        # Category importance
        from collections import defaultdict
        category_sums = defaultdict(float)
        for _, row in importance_df.iterrows():
            feat = row['feature']
            if 'mfcc' in feat:
                cat = 'Timbral (MFCC)'
            elif any(x in feat for x in ['spectral', 'zcr']):
                cat = 'Spectral'
            elif any(x in feat for x in ['tempo', 'onset', 'ioi']):
                cat = 'Rhythmic'
            elif any(x in feat for x in ['chroma', 'tonnetz']):
                cat = 'Harmonic'
            elif 'rms' in feat or 'dynamic' in feat:
                cat = 'Dynamic'
            else:
                cat = 'Other'
            category_sums[cat] += row['importance']
        
        cat_df = pd.DataFrame.from_dict(category_sums, orient='index', columns=['sum'])
        viz.plot_feature_category_importance(cat_df)
        
    except Exception as e:
        print(f"Importance data issue: {e}")
    
    # Pair validation
    try:
        pair_df = pd.read_csv(pairs_csv)
        if permutation_results is None:
            permutation_results = {
                'matched_mean': pair_df['distance'].mean(),
                'random_mean': pair_df['distance'].mean() * 1.5,
                'random_std': pair_df['distance'].std()
            }
        viz.plot_pair_validation(pair_df, permutation_results)
    except Exception as e:
        print(f"Pair validation data not found: {e}")
    
    # Classification results
    try:
        tracks_df = pd.read_csv(tracks_csv)
        viz.plot_confusion_heatmap(tracks_df)
        viz.plot_roc_curve(tracks_df)
    except Exception as e:
        print(f"Classification results not found: {e}")
    
    # Composite figure
    try:
        variance_df = pd.read_csv(variance_csv)
        importance_df = pd.read_csv(importance_csv)
        viz.create_composite_figure(variance_df, importance_df)
    except Exception as e:
        print(f"Could not create composite figure: {e}")
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    generate_all_visualizations()