"""
Data Preprocessing Module for Astronomical Object Classification
Handles data loading, cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class AstronomicalDataProcessor:
    """
    Advanced data processor for astronomical datasets with specialized feature engineering.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, file_path):
        """
        Load astronomical data from various formats.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            print(f"✅ Data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """
        Comprehensive data exploration and statistics.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        print("🔍 DATA EXPLORATION REPORT")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset Shape: {data.shape}")
        print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n📊 Data Types:")
        print(data.dtypes.value_counts())
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            print("\n❌ Missing Values:")
            print(missing_data[missing_data > 0])
        else:
            print("\n✅ No missing values found!")
            
        # Target distribution
        if 'class' in data.columns:
            print(f"\n🎯 Target Distribution:")
            print(data['class'].value_counts())
            print(f"Class Balance: {data['class'].value_counts(normalize=True).round(3)}")
        
        # Numerical features statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 Numerical Features Statistics:")
            print(data[numerical_cols].describe())
    
    def clean_data(self, data):
        """
        Clean and preprocess the astronomical data.
        
        Args:
            data (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("🧹 Cleaning data...")
        data_clean = data.copy()
        
        # Remove duplicates
        initial_rows = len(data_clean)
        data_clean = data_clean.drop_duplicates()
        if len(data_clean) < initial_rows:
            print(f"   Removed {initial_rows - len(data_clean)} duplicate rows")
        
        # Handle missing values
        data_clean = self._handle_missing_values(data_clean)
        
        # Remove outliers using IQR method for numerical columns
        data_clean = self._remove_outliers(data_clean)
        
        print(f"✅ Data cleaning completed. Shape: {data_clean.shape}")
        return data_clean
    
    def _handle_missing_values(self, data):
        """Handle missing values using advanced imputation strategies."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # For numerical columns, use KNN imputation for better accuracy
        if len(numerical_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            data[numerical_cols] = knn_imputer.fit_transform(data[numerical_cols])
        
        # For categorical columns, use mode imputation
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def _remove_outliers(self, data, threshold=1.5):
        """Remove outliers using IQR method."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Replace outliers with bounds instead of removing
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        return data
    
    def engineer_features(self, data):
        """
        Create advanced astronomical features for better classification.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("⚙️ Engineering astronomical features...")
        data_eng = data.copy()
        
        # Astronomical color indices (if u, g, r, i, z bands exist)
        color_bands = ['u', 'g', 'r', 'i', 'z']
        available_bands = [col for col in color_bands if col in data_eng.columns]
        
        if len(available_bands) >= 2:
            # Create color indices (magnitude differences)
            for i in range(len(available_bands) - 1):
                for j in range(i + 1, len(available_bands)):
                    band1, band2 = available_bands[i], available_bands[j]
                    if band1 in data_eng.columns and band2 in data_eng.columns:
                        color_index = f"{band1}_{band2}_color"
                        data_eng[color_index] = data_eng[band1] - data_eng[band2]
                        print(f"   Created color index: {color_index}")
        
        # Flux ratios and other astronomical features
        if 'redshift' in data_eng.columns:
            # Redshift-based features
            data_eng['redshift_squared'] = data_eng['redshift'] ** 2
            data_eng['redshift_log'] = np.log1p(data_eng['redshift'])
        
        # Magnitude-based features
        magnitude_cols = [col for col in data_eng.columns if 'mag' in col.lower()]
        if len(magnitude_cols) > 0:
            data_eng['mean_magnitude'] = data_eng[magnitude_cols].mean(axis=1)
            data_eng['magnitude_std'] = data_eng[magnitude_cols].std(axis=1)
        
        # Statistical features for numerical columns
        numerical_cols = data_eng.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Create interaction features
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    if col1 != col2:
                        interaction_name = f"{col1}_{col2}_interaction"
                        data_eng[interaction_name] = data_eng[col1] * data_eng[col2]
        
        print(f"✅ Feature engineering completed. New shape: {data_eng.shape}")
        return data_eng
    
    def prepare_features(self, data, target_col='class'):
        """
        Prepare features and target for machine learning.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X, y) features and target
        """
        print("🎯 Preparing features for ML...")
        
        # Separate features and target
        if target_col in data.columns:
            y = data[target_col]
            X = data.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Encode categorical target
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print(f"   Encoded target classes: {self.label_encoder.classes_}")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            print(f"   One-hot encoded {len(categorical_cols)} categorical features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Feature preparation completed. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y
    
    def scale_features(self, X, method='standard'):
        """
        Scale features using various methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            method (str): Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            pd.DataFrame: Scaled features
        """
        print(f"📏 Scaling features using {method} method...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling method")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print("✅ Feature scaling completed")
        return X_scaled
    
    def select_features(self, X, y, method='mutual_info', k=None):
        """
        Select the most important features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target variable
            method (str): Feature selection method
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        if k is None:
            k = min(X.shape[1], 50)  # Default to 50 features or all if less
        
        print(f"🎯 Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_score':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError("Unsupported feature selection method")
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        print(f"✅ Feature selection completed. Selected {len(selected_features)} features")
        return X_selected
    
    def get_feature_importance(self):
        """Get feature importance scores if available."""
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            selected_features = self.feature_names[self.feature_selector.get_support()]
            
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance_score': scores[self.feature_selector.get_support()]
            }).sort_values('importance_score', ascending=False)
            
            return importance_df
        
        return None
    
    def save_processor(self, filepath):
        """Save the fitted processor for later use."""
        import joblib
        joblib.dump(self, filepath)
        print(f"💾 Processor saved to {filepath}")
    
    @classmethod
    def load_processor(cls, filepath):
        """Load a saved processor."""
        import joblib
        return joblib.load(filepath)
