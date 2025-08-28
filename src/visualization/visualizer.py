"""
Visualization Module for Astronomical Object Classification
Provides comprehensive plotting and visualization tools for data analysis and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AstronomicalVisualizer:
    """
    Comprehensive visualizer for astronomical data and ML results.
    """
    
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.astronomical_colors = ['#FFD700', '#4169E1', '#DC143C']  # Gold, Blue, Red
        
    def plot_data_distribution(self, data, target_col='class', figsize=(15, 10)):
        """
        Plot comprehensive data distribution analysis.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("📊 Creating data distribution plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Astronomical Data Distribution Analysis', fontsize=16)
        
        # Target distribution
        if target_col in data.columns:
            target_counts = data[target_col].value_counts()
            axes[0, 0].pie(target_counts.values, labels=target_counts.index, 
                          autopct='%1.1f%%', colors=self.astronomical_colors)
            axes[0, 0].set_title('Target Distribution')
        
        # Numerical features distribution
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Select first few numerical columns for visualization
            plot_cols = numerical_cols[:4]
            
            for i, col in enumerate(plot_cols):
                row = i // 2
                col_idx = i % 2
                
                if row < 2 and col_idx < 3:
                    axes[row, col_idx + 1].hist(data[col].dropna(), bins=30, alpha=0.7, 
                                              color=self.colors[i % len(self.colors)])
                    axes[row, col_idx + 1].set_title(f'{col} Distribution')
                    axes[row, col_idx + 1].set_xlabel(col)
                    axes[row, col_idx + 1].set_ylabel('Frequency')
        
        # Missing values heatmap
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            missing_pct = (missing_data / len(data)) * 100
            axes[1, 2].bar(range(len(missing_pct)), missing_pct, 
                          color=['red' if x > 0 else 'green' for x in missing_pct])
            axes[1, 2].set_title('Missing Values (%)')
            axes[1, 2].set_xlabel('Features')
            axes[1, 2].set_ylabel('Missing %')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data, figsize=(12, 10)):
        """
        Plot correlation matrix for numerical features.
        
        Args:
            data (pd.DataFrame): Input dataset
            figsize (tuple): Figure size
        """
        print("🔗 Creating correlation matrix...")
        
        # Select numerical columns
        numerical_data = data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) < 2:
            print("⚠️ Need at least 2 numerical columns for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance from trained models.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        print("🎯 Creating feature importance plot...")
        
        if importance_df is None or len(importance_df) == 0:
            print("⚠️ No feature importance data available")
            return
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[:len(top_features)])
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, results_dict, figsize=(20, 15)):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            results_dict (dict): Dictionary of model results
            figsize (tuple): Figure size
        """
        print("📈 Creating confusion matrices...")
        
        n_models = len(results_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        class_labels = ['Star', 'Galaxy', 'Quasar']
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            row = i // n_cols
            col = i % n_cols
            
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_labels, yticklabels=class_labels,
                           ax=axes[row, col])
                axes[row, col].set_title(f'{model_name.upper()}\nConfusion Matrix')
                axes[row, col].set_xlabel('Predicted Label')
                axes[row, col].set_ylabel('True Label')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, results_dict, figsize=(12, 8)):
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_dict (dict): Dictionary of model results
            figsize (tuple): Figure size
        """
        print("📊 Creating ROC curves...")
        
        plt.figure(figsize=figsize)
        
        for model_name, results in results_dict.items():
            if results.get('has_probabilities') and results.get('probabilities') is not None:
                y_proba = results['probabilities']
                y_test = results.get('y_test', None)
                
                if y_test is not None and y_proba.shape[1] == 2:
                    from sklearn.metrics import roc_curve, roc_auc_score
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance_comparison(self, comparison_df, figsize=(15, 10)):
        """
        Create comprehensive model performance comparison.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            figsize (tuple): Figure size
        """
        print("🏆 Creating model performance comparison...")
        
        if comparison_df is None or len(comparison_df) == 0:
            print("⚠️ No comparison data available")
            return
        
        # Metrics to plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa', 'ROC-AUC']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(available_metrics):
            row = i // n_cols
            col = i % n_cols
            
            # Filter out NaN values
            data = comparison_df.dropna(subset=[metric])
            
            if len(data) > 0:
                bars = axes[row, col].bar(data['Model'], data[metric], 
                                        color=self.colors[:len(data)])
                axes[row, col].set_title(f'{metric} Comparison')
                axes[row, col].set_ylabel(metric)
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}', ha='center', va='bottom', 
                                      fontweight='bold')
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_astronomical_features(self, data, feature_cols=None, target_col='class', figsize=(15, 10)):
        """
        Plot astronomical-specific features and relationships.
        
        Args:
            data (pd.DataFrame): Input dataset
            feature_cols (list): List of feature columns to plot
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("🌟 Creating astronomical feature plots...")
        
        if feature_cols is None:
            # Default astronomical features
            feature_cols = ['u', 'g', 'r', 'i', 'z', 'redshift']
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        if len(feature_cols) < 2:
            print("⚠️ Need at least 2 feature columns for astronomical plots")
            return
        
        # Create subplots
        n_features = len(feature_cols)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Astronomical Features Analysis', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each feature
        for i, feature in enumerate(feature_cols):
            row = i // n_cols
            col = i % n_cols
            
            if feature in data.columns:
                # Create scatter plot with target coloring
                if target_col in data.columns:
                    scatter = axes[row, col].scatter(data.index, data[feature], 
                                                   c=data[target_col].astype('category').cat.codes,
                                                   cmap='viridis', alpha=0.6)
                    axes[row, col].set_title(f'{feature} vs Index')
                    axes[row, col].set_xlabel('Object Index')
                    axes[row, col].set_ylabel(feature)
                else:
                    axes[row, col].hist(data[feature].dropna(), bins=30, alpha=0.7)
                    axes[row, col].set_title(f'{feature} Distribution')
                    axes[row, col].set_xlabel(feature)
                    axes[row, col].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimensionality_reduction(self, X, y=None, method='pca', figsize=(15, 5)):
        """
        Plot dimensionality reduction results (PCA, t-SNE).
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target variable (optional)
            method (str): Reduction method ('pca' or 'tsne')
            figsize (tuple): Figure size
        """
        print(f"🔍 Creating {method.upper()} visualization...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'{method.upper()} Dimensionality Reduction Analysis', fontsize=16)
        
        if method.lower() == 'pca':
            # PCA analysis
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            axes[0].plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
            axes[0].set_title('Explained Variance Ratio')
            axes[0].set_xlabel('Principal Component')
            axes[0].set_ylabel('Explained Variance Ratio')
            axes[0].grid(True)
            
            axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
            axes[1].set_title('Cumulative Explained Variance')
            axes[1].set_xlabel('Principal Component')
            axes[1].set_ylabel('Cumulative Explained Variance')
            axes[1].grid(True)
            
            # First two components
            if y is not None:
                scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
                axes[2].set_title('First Two Principal Components')
                axes[2].set_xlabel('PC1')
                axes[2].set_ylabel('PC2')
                axes[2].grid(True)
        
        elif method.lower() == 'tsne':
            # t-SNE analysis
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_scaled)
            
            if y is not None:
                scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
                axes[0].set_title('t-SNE 2D Projection')
                axes[0].set_xlabel('t-SNE 1')
                axes[0].set_ylabel('t-SNE 2')
                axes[0].grid(True)
            
            # Hide unused subplots
            axes[1].set_visible(False)
            axes[2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, data, target_col='class'):
        """
        Create interactive Plotly dashboard for data exploration.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        print("🎛️ Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Target Distribution', 'Feature Correlation', 'Feature Distribution', 'Missing Values'],
            specs=[[{"type": "pie"}, {"type": "heatmap"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Target distribution
        if target_col in data.columns:
            target_counts = data[target_col].value_counts()
            fig.add_trace(
                go.Pie(labels=target_counts.index, values=target_counts.values, name="Target"),
                row=1, col=1
            )
        
        # Feature correlation
        numerical_data = data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) >= 2:
            corr_matrix = numerical_data.corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=1, col=2
            )
        
        # Feature distribution (first numerical feature)
        if len(numerical_data.columns) > 0:
            first_feature = numerical_data.columns[0]
            fig.add_trace(
                go.Histogram(x=data[first_feature], name=first_feature),
                row=2, col=1
            )
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Values"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Astronomical Data Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def save_plots(self, filename='astronomical_plots.png', dpi=300):
        """
        Save current plots to file.
        
        Args:
            filename (str): Output filename
            dpi (int): DPI for saved image
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"💾 Plots saved to {filename}")
    
    def set_style(self, style='default'):
        """
        Set plotting style.
        
        Args:
            style (str): Style name ('default', 'dark', 'minimal')
        """
        if style == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        elif style == 'minimal':
            plt.style.use('default')
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
        else:
            plt.style.use('seaborn-v0_8')
        
        print(f"🎨 Plotting style set to '{style}'")
