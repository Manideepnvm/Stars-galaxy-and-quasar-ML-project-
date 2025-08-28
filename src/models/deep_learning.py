"""
Deep Learning Models for Astronomical Object Classification
Implements CNN, MLP, and RNN architectures with advanced training techniques.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DeepLearningTrainer:
    """
    Comprehensive deep learning trainer with multiple architectures.
    """
    
    def __init__(self, input_shape=None, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.histories = {}
        self.scalers = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def create_mlp(self, input_dim, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        """
        Create Multi-Layer Perceptron (MLP) model.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            keras.Model: Compiled MLP model
        """
        print("🧠 Creating MLP model...")
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ MLP created with {model.count_params():,} parameters")
        return model
    
    def create_cnn(self, input_shape, filters=[32, 64, 128], kernel_sizes=[3, 3, 3]):
        """
        Create Convolutional Neural Network (CNN) model.
        
        Args:
            input_shape (tuple): Shape of input data
            filters (list): List of filter sizes for each conv layer
            kernel_sizes (list): List of kernel sizes for each conv layer
            
        Returns:
            keras.Model: Compiled CNN model
        """
        print("🖼️ Creating CNN model...")
        
        model = models.Sequential()
        
        # Reshape input for 1D convolution (for tabular data)
        if len(input_shape) == 1:
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            input_shape = (input_shape[0], 1)
        
        # Convolutional layers
        for i, (filters_num, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            if i == 0:
                model.add(layers.Conv1D(filters_num, kernel_size, activation='relu', input_shape=input_shape))
            else:
                model.add(layers.Conv1D(filters_num, kernel_size, activation='relu'))
            
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Dropout(0.3))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ CNN created with {model.count_params():,} parameters")
        return model
    
    def create_rnn(self, input_shape, lstm_units=[128, 64], dropout_rate=0.3):
        """
        Create Recurrent Neural Network (RNN) model.
        
        Args:
            input_shape (tuple): Shape of input data
            lstm_units (list): List of LSTM units for each layer
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            keras.Model: Compiled RNN model
        """
        print("🔄 Creating RNN model...")
        
        model = models.Sequential()
        
        # Reshape input for LSTM (for tabular data)
        if len(input_shape) == 1:
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            input_shape = (input_shape[0], 1)
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences, dropout=dropout_rate))
            model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ RNN created with {model.count_params():,} parameters")
        return model
    
    def create_ensemble(self, models_list, input_shape):
        """
        Create ensemble model combining multiple architectures.
        
        Args:
            models_list (list): List of model creation functions
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled ensemble model
        """
        print("🎭 Creating ensemble model...")
        
        # Create individual models
        individual_models = []
        for model_func in models_list:
            if 'cnn' in str(model_func).lower():
                model = self.create_cnn(input_shape)
            elif 'rnn' in str(model_func).lower():
                model = self.create_rnn(input_shape)
            else:
                model = self.create_mlp(input_shape[0] if len(input_shape) == 1 else input_shape[0])
            individual_models.append(model)
        
        # Create ensemble input
        ensemble_input = layers.Input(shape=input_shape)
        
        # Get predictions from each model
        ensemble_outputs = []
        for model in individual_models:
            # Freeze the model weights
            model.trainable = False
            output = model(ensemble_input)
            ensemble_outputs.append(output)
        
        # Average the predictions
        ensemble_output = layers.Average()(ensemble_outputs)
        
        # Create ensemble model
        ensemble_model = models.Model(inputs=ensemble_input, outputs=ensemble_output)
        
        # Compile ensemble
        ensemble_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Ensemble created with {ensemble_model.count_params():,} parameters")
        return ensemble_model
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Prepare data for deep learning models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target variable
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("📊 Preparing data for deep learning...")
        
        # Convert to numpy arrays
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['standard'] = scaler
        
        # Convert to categorical for multi-class
        if self.num_classes > 2:
            y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
            y_test_cat = y_test
        
        print(f"✅ Data prepared. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train_cat, y_val_cat, y_test_cat)
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name='model', epochs=100, batch_size=32):
        """
        Train a deep learning model.
        
        Args:
            model (keras.Model): Model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name (str): Name for the model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.Model: Trained model
        """
        print(f"🚀 Training {model_name}...")
        
        # Callbacks for better training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        print(f"✅ {model_name} training completed!")
        return model
    
    def train_all_models(self, X, y, epochs=100, batch_size=32):
        """
        Train all available deep learning models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target variable
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Dictionary of trained models
        """
        print("🚀 Training all deep learning models...")
        print("=" * 50)
        
        # Prepare data
        (X_train, X_val, X_test, 
         y_train, y_val, y_test) = self.prepare_data(X, y)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Train MLP
        mlp_model = self.create_mlp(input_shape[0] if len(input_shape) == 1 else input_shape[0])
        self.train_model(mlp_model, X_train, y_train, X_val, y_val, 
                        'mlp', epochs, batch_size)
        
        # Train CNN
        cnn_model = self.create_cnn(input_shape)
        self.train_model(cnn_model, X_train, y_train, X_val, y_val, 
                        'cnn', epochs, batch_size)
        
        # Train RNN
        rnn_model = self.create_rnn(input_shape)
        self.train_model(rnn_model, X_train, y_train, X_val, y_val, 
                        'rnn', epochs, batch_size)
        
        # Train Ensemble
        ensemble_model = self.create_ensemble([self.create_mlp, self.create_cnn, self.create_rnn], input_shape)
        self.train_model(ensemble_model, X_train, y_train, X_val, y_val, 
                        'ensemble', epochs, batch_size)
        
        print("\n🎉 All deep learning models trained successfully!")
        return self.models
    
    def evaluate_model(self, model_name):
        """
        Evaluate a specific model on test data.
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            dict: Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate metrics
        accuracy = (y_pred_classes == y_test_classes).mean()
        
        # Classification report
        report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(self.y_test, y_pred, multi_class='ovr')
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
    
    def plot_training_history(self, model_name):
        """
        Plot training history for a specific model.
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.histories:
            print(f"⚠️ No training history found for {model_name}")
            return
        
        history = self.histories[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name.upper()}', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, directory='models/'):
        """Save all trained models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}.h5')
            model.save(filepath)
            print(f"💾 Saved {name} to {filepath}")
    
    def load_models(self, directory='models/'):
        """Load saved models."""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('.h5'):
                name = filename.replace('.h5', '')
                filepath = os.path.join(directory, filename)
                self.models[name] = keras.models.load_model(filepath)
                print(f"📂 Loaded {name} from {filepath}")
