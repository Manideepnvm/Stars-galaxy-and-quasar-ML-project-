"""
Machine Learning Models for Astronomical Object Classification
Implements various ML algorithms with hyperparameter tuning and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """
    Comprehensive ML model trainer with multiple algorithms and hyperparameter tuning.
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def train_random_forest(self, X, y, cv=5, n_jobs=-1):
        """Train Random Forest with hyperparameter tuning."""
        print("🌲 Training Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['random_forest'] = grid_search.best_estimator_
        self.best_models['random_forest'] = grid_search.best_estimator_
        self.cv_scores['random_forest'] = grid_search.best_score_
        
        # Feature importance
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X.columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Random Forest trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_xgboost(self, X, y, cv=5, n_jobs=-1):
        """Train XGBoost with hyperparameter tuning."""
        print("🚀 Training XGBoost...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=n_jobs)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['xgboost'] = grid_search.best_estimator_
        self.best_models['xgboost'] = grid_search.best_estimator_
        self.cv_scores['xgboost'] = grid_search.best_score_
        
        # Feature importance
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X.columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ XGBoost trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X, y, cv=5, n_jobs=-1):
        """Train LightGBM with hyperparameter tuning."""
        print("💡 Training LightGBM...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 62, 127],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=n_jobs, verbose=-1)
        grid_search = GridSearchCV(lgb_model, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['lightgbm'] = grid_search.best_estimator_
        self.best_models['lightgbm'] = grid_search.best_estimator_
        self.cv_scores['lightgbm'] = grid_search.best_score_
        
        # Feature importance
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X.columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ LightGBM trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_svm(self, X, y, cv=5, n_jobs=-1):
        """Train Support Vector Machine with hyperparameter tuning."""
        print("🔧 Training SVM...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Grid search with cross-validation
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['svm'] = grid_search.best_estimator_
        self.best_models['svm'] = grid_search.best_estimator_
        self.cv_scores['svm'] = grid_search.best_score_
        
        print(f"✅ SVM trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_logistic_regression(self, X, y, cv=5, n_jobs=-1):
        """Train Logistic Regression with hyperparameter tuning."""
        print("📊 Training Logistic Regression...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search with cross-validation
        lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=n_jobs)
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['logistic_regression'] = grid_search.best_estimator_
        self.best_models['logistic_regression'] = grid_search.best_estimator_
        self.cv_scores['logistic_regression'] = grid_search.best_score_
        
        print(f"✅ Logistic Regression trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_knn(self, X, y, cv=5, n_jobs=-1):
        """Train K-Nearest Neighbors with hyperparameter tuning."""
        print("🎯 Training KNN...")
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # Grid search with cross-validation
        knn = KNeighborsClassifier(n_jobs=n_jobs)
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        # Store results
        self.models['knn'] = grid_search.best_estimator_
        self.best_models['knn'] = grid_search.best_estimator_
        self.cv_scores['knn'] = grid_search.best_score_
        
        print(f"✅ KNN trained. Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_ensemble(self, X, y, cv=5, n_jobs=-1):
        """Train ensemble model combining best models."""
        print("🎭 Training Ensemble Model...")
        
        # Get best models for ensemble
        best_models = []
        for name, model in self.best_models.items():
            if hasattr(model, 'predict_proba'):
                best_models.append((name, model))
        
        if len(best_models) < 2:
            print("⚠️ Need at least 2 models with predict_proba for ensemble")
            return None
        
        # Create voting classifier
        estimators = [(name, model) for name, model in best_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        ensemble.fit(X, y)
        
        # Store results
        self.models['ensemble'] = ensemble
        self.best_models['ensemble'] = ensemble
        self.cv_scores['ensemble'] = cv_scores.mean()
        
        print(f"✅ Ensemble trained. CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return ensemble
    
    def train_all_models(self, X, y, cv=5, n_jobs=-1):
        """Train all available models."""
        print("🚀 Training all ML models...")
        print("=" * 50)
        
        # Train each model
        self.train_random_forest(X, y, cv, n_jobs)
        self.train_xgboost(X, y, cv, n_jobs)
        self.train_lightgbm(X, y, cv, n_jobs)
        self.train_svm(X, y, cv, n_jobs)
        self.train_logistic_regression(X, y, cv, n_jobs)
        self.train_knn(X, y, cv, n_jobs)
        
        # Train ensemble
        self.train_ensemble(X, y, cv, n_jobs)
        
        print("\n🎉 All models trained successfully!")
        return self.models
    
    def get_model_comparison(self):
        """Get comparison of all trained models."""
        if not self.cv_scores:
            return None
        
        comparison_df = pd.DataFrame({
            'Model': list(self.cv_scores.keys()),
            'CV_Score': list(self.cv_scores.values())
        }).sort_values('CV_Score', ascending=False)
        
        return comparison_df
    
    def get_best_model(self):
        """Get the best performing model."""
        if not self.cv_scores:
            return None
        
        best_model_name = max(self.cv_scores, key=self.cv_scores.get)
        return self.best_models[best_model_name], best_model_name
    
    def save_models(self, directory='models/'):
        """Save all trained models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}.joblib')
            joblib.dump(model, filepath)
            print(f"💾 Saved {name} to {filepath}")
    
    def load_models(self, directory='models/'):
        """Load saved models."""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('.joblib'):
                name = filename.replace('.joblib', '')
                filepath = os.path.join(directory, filename)
                self.models[name] = joblib.load(filepath)
                print(f"📂 Loaded {name} from {filepath}")
    
    def predict(self, model_name, X):
        """Make predictions using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """Get prediction probabilities using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(X)
        else:
            raise ValueError(f"Model '{model_name}' doesn't support predict_proba")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model on test data."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }
        
        return results
