#!/usr/bin/env python3
"""
LSTM-based Predictive Congestion Agent (PCA)
This module implements an LSTM neural network for predicting traffic congestion
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
from datetime import datetime


class LSTMCongestionPredictor:
    """
    LSTM-based model for predicting traffic congestion
    """
    
    def __init__(self, sequence_length=16, prediction_horizon=15, features=None):
        """
        Initialize the LSTM predictor
        
        Args:
            sequence_length: Number of historical time steps to use (default: 16 = 4 hours at 15-min intervals)
            prediction_horizon: Minutes ahead to predict (default: 15)
            features: List of feature names
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or ['flow_rate', 'avg_speed', 'hour', 'day_of_week']
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def prepare_data(self, df, target_column='congestion_risk_score'):
        """
        Prepare data for LSTM training
        
        Args:
            df: DataFrame with traffic data
            target_column: Name of the target column
            
        Returns:
            X, y: Prepared sequences and targets
        """
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Extract time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Calculate congestion risk score if not present
        if target_column not in df.columns:
            # Simple heuristic: high flow rate + low speed = high congestion
            df['congestion_risk_score'] = self._calculate_congestion_score(df)
        
        # Select features
        feature_data = df[self.features].values
        target_data = df[target_column].values
        
        # Normalize features
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data_scaled) - self.sequence_length):
            X.append(feature_data_scaled[i:i + self.sequence_length])
            y.append(target_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def _calculate_congestion_score(self, df):
        """
        Calculate congestion risk score from flow rate and speed
        
        Args:
            df: DataFrame with 'flow_rate' and 'avg_speed' columns
            
        Returns:
            Congestion risk score (0-1)
        """
        # Normalize flow rate and speed
        flow_norm = (df['flow_rate'] - df['flow_rate'].min()) / (df['flow_rate'].max() - df['flow_rate'].min() + 1e-6)
        speed_norm = (df['avg_speed'] - df['avg_speed'].min()) / (df['avg_speed'].max() - df['avg_speed'].min() + 1e-6)
        
        # High flow + low speed = high congestion
        congestion_score = flow_norm * (1 - speed_norm)
        
        return congestion_score
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
        """
        model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer with return sequences
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            # Third LSTM layer
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer (congestion risk score: 0-1)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print("\nTraining LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed")
    
    def predict(self, X):
        """
        Predict congestion risk score
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted congestion risk scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_from_history(self, recent_data):
        """
        Predict congestion from recent historical data
        
        Args:
            recent_data: DataFrame with recent traffic data (at least sequence_length rows)
            
        Returns:
            Predicted congestion risk score
        """
        # Prepare data
        feature_data = recent_data[self.features].values
        
        # Take last sequence_length samples
        if len(feature_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples")
        
        feature_data = feature_data[-self.sequence_length:]
        
        # Normalize
        feature_data_scaled = self.scaler.transform(feature_data)
        
        # Reshape for prediction
        X = feature_data_scaled.reshape(1, self.sequence_length, len(self.features))
        
        # Predict
        prediction = self.predict(X)
        
        return prediction[0][0]
    
    def save(self, model_dir):
        """
        Save model and scaler
        
        Args:
            model_dir: Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features
        }
        config_path = os.path.join(model_dir, 'config.pkl')
        joblib.dump(config, config_path)
        
        print(f"\n✓ Model saved to {model_dir}")
    
    def load(self, model_dir):
        """
        Load model and scaler
        
        Args:
            model_dir: Directory containing model files
        """
        # Load model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        # Load configuration
        config_path = os.path.join(model_dir, 'config.pkl')
        config = joblib.load(config_path)
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.features = config['features']
        
        print(f"✓ Model loaded from {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Congestion Predictor')
    parser.add_argument('--data', required=True, help='Path to traffic data CSV file')
    parser.add_argument('--output', default='../models/lstm_pca', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Initialize predictor
    predictor = LSTMCongestionPredictor()
    
    # Prepare data
    print("Preparing data...")
    X, y = predictor.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    predictor.train(X_train, y_train, X_val, y_val, 
                   epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae, test_mse = predictor.model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model
    predictor.save(args.output)


if __name__ == "__main__":
    main()
