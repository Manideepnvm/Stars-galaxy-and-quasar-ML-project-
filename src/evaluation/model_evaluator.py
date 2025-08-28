"""
Model Evaluation Module for Astronomical Object Classification
Provides comprehensive evaluation metrics, analysis, and comparison tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluator with advanced metrics and visualization.
    """
    
    def __init__(self):
        self.results = {}
        self.comparison_df = None
        
    def evaluate_classification_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate a classification model comprehensively.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: True test labels
            model_name: Name of the model for identification
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print(f"🔍 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test)
            has_proba = True
        except:
            y_proba = None
            has_proba = False
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if has_proba and y_proba is not None:
            try:
                if y_proba.shape[1] == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except:
                roc_auc = None
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'kappa': kappa,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
            'has_probabilities': has_proba
        }
        
        self.results[model_name] = results
        
        print(f"✅ {model_name} evaluation completed!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"   ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        print(f"🔄 Performing {cv}-fold cross-validation...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Calculate statistics
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        cv_min = cv_scores.min()
        cv_max = cv_scores.max()
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean': cv_mean,
            'std': cv_std,
            'min': cv_min,
            'max': cv_max,
            'folds': cv
        }
        
        print(f"✅ Cross-validation completed!")
        print(f"   Mean {scoring}: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Range: [{cv_min:.4f}, {cv_max:.4f}]")
        
        return cv_results
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare multiple models and create comparison dataframe.
        
        Args:
            models_dict: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: True test labels
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        print("📊 Comparing all models...")
        
        comparison_results = []
        
        for name, model in models_dict.items():
            # Evaluate model
            results = self.evaluate_classification_model(model, X_test, y_test, name)
            
            # Extract key metrics
            comparison_results.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Kappa': results['kappa'],
                'ROC-AUC': results['roc_auc'] if results['roc_auc'] else np.nan
            })
        
        # Create comparison dataframe
        self.comparison_df = pd.DataFrame(comparison_results)
        self.comparison_df = self.comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n🏆 MODEL COMPARISON RESULTS")
        print("=" * 50)
        print(self.comparison_df.round(4))
        
        return self.comparison_df
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model to plot
            figsize: Figure size
        """
        if model_name not in self.results:
            print(f"⚠️ No results found for {model_name}")
            return
        
        results = self.results[model_name]
        cm = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Star', 'Galaxy', 'Quasar'],
                   yticklabels=['Star', 'Galaxy', 'Quasar'])
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """
        Plot ROC curves for all models with probabilities.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            if results['has_probabilities'] and results['probabilities'] is not None:
                y_proba = results['probabilities']
                y_test = results.get('y_test', None)
                
                if y_test is not None and y_proba.shape[1] == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self, figsize=(12, 8)):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            if results['has_probabilities'] and results['probabilities'] is not None:
                y_proba = results['probabilities']
                y_test = results.get('y_test', None)
                
                if y_test is not None and y_proba.shape[1] == 2:
                    # Binary classification
                    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                    plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, figsize=(15, 10)):
        """
        Create comprehensive model comparison visualization.
        
        Args:
            figsize: Figure size
        """
        if self.comparison_df is None:
            print("⚠️ No comparison data available. Run compare_models() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Metrics to plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa', 'ROC-AUC']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            if metric in self.comparison_df.columns:
                # Filter out NaN values
                data = self.comparison_df.dropna(subset=[metric])
                
                if len(data) > 0:
                    bars = axes[row, col].bar(data['Model'], data[metric])
                    axes[row, col].set_title(f'{metric} Comparison')
                    axes[row, col].set_ylabel(metric)
                    axes[row, col].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                          f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_comparison(self):
        """
        Create interactive Plotly comparison dashboard.
        
        Returns:
            plotly.graph_objects.Figure: Interactive comparison plot
        """
        if self.comparison_df is None:
            print("⚠️ No comparison data available. Run compare_models() first.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa', 'ROC-AUC'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces for each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa', 'ROC-AUC']
        
        for i, metric in enumerate(metrics):
            if metric in self.comparison_df.columns:
                # Filter out NaN values
                data = self.comparison_df.dropna(subset=[metric])
                
                if len(data) > 0:
                    row = (i // 3) + 1
                    col = (i % 3) + 1
                    
                    fig.add_trace(
                        go.Bar(
                            x=data['Model'],
                            y=data[metric],
                            name=metric,
                            text=[f'{val:.3f}' for val in data[metric]],
                            textposition='auto'
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Model Performance Comparison",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def generate_evaluation_report(self, output_file=None):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            str: Report content
        """
        if not self.results:
            return "No evaluation results available."
        
        report = []
        report.append("=" * 60)
        report.append("ASTRONOMICAL OBJECT CLASSIFICATION - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall comparison
        if self.comparison_df is not None:
            report.append("🏆 OVERALL MODEL RANKING")
            report.append("-" * 30)
            report.append(self.comparison_df.to_string(index=False))
            report.append("")
        
        # Individual model details
        for model_name, results in self.results.items():
            report.append(f"📊 DETAILED RESULTS - {model_name.upper()}")
            report.append("-" * 40)
            report.append(f"Accuracy: {results['accuracy']:.4f}")
            report.append(f"Precision: {results['precision']:.4f}")
            report.append(f"Recall: {results['recall']:.4f}")
            report.append(f"F1-Score: {results['f1_score']:.4f}")
            report.append(f"Kappa: {results['kappa']:.4f}")
            if results['roc_auc']:
                report.append(f"ROC-AUC: {results['roc_auc']:.4f}")
            report.append("")
            
            # Classification report
            if 'classification_report' in results:
                report.append("Classification Report:")
                report.append(str(pd.DataFrame(results['classification_report']).round(4)))
                report.append("")
        
        report_content = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"💾 Evaluation report saved to {output_file}")
        
        return report_content
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if self.comparison_df is None:
            return None, None
        
        if metric not in self.comparison_df.columns:
            print(f"⚠️ Metric '{metric}' not available")
            return None, None
        
        # Find best model
        best_idx = self.comparison_df[metric].idxmax()
        best_model = self.comparison_df.loc[best_idx, 'Model']
        best_score = self.comparison_df.loc[best_idx, metric]
        
        return best_model, best_score
