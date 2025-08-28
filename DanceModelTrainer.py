import json
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
import tensorboard

from PoseProcessing import PoseAnalyzer
from PoseProcessing import EnhancedPoseAnalyzer
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import TensorBoard

import logging

logger = logging.getLogger(__name__)

class DanceModelTrainer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.model_dir = "dance_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, song_name):
        # Load reference (professional) performances
        X_ref = self.data_manager.load_reference_data(song_name)
        
        if not X_ref:
            logger.warning(f"No reference data found for {song_name}")
            return [], np.array([])
        
        # Create labels (perfect performance)
        y = [1] * len(X_ref)  # 1 = perfect performance
        
        # Add negative examples if available
        user_data = self.data_manager.load_user_data(song_name)
        if user_data:
            X_ref.extend(user_data)
            y.extend([0] * len(user_data))  # 0 = imperfect
        
        # Ensure all sequences have the same number of features
        X_clean = []
        y_clean = []
        for i, seq in enumerate(X_ref):
            # Filter out any sequences with incorrect dimensions
            if seq and all(len(frame) == len(PoseAnalyzer.ANGLE_ORDER) for frame in seq):
                X_clean.append(seq)
                y_clean.append(y[i])
            else:
                logger.warning(f"Skipping invalid sequence with shape {[len(frame) for frame in seq] if seq else 'empty'}")
        
        return X_clean, np.array(y_clean)
    
    def train_model(self, song_name, sequence_length=30):
        try:
            X, y = self.prepare_data(song_name)
            
            if not X:
                raise ValueError(f"No valid training data found for {song_name}")
            
            # Handle cases with insufficient data
            if len(X) == 1:
                # Only one sample, use it for training and create a small validation set
                X_train, y_train = X, y
                X_test, y_test = X, y
                logger.warning("Only one sample available. Using it for both training and validation.")
            elif len(X) <= 3:
                # Very few samples, use a smaller test size
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=42
                )
            else:
                # Enough samples, use standard split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Pad sequences to uniform length
            X_train = self.pad_sequences(X_train, sequence_length)
            X_test = self.pad_sequences(X_test, sequence_length)
            
            # Build LSTM model with Input layer
            model = Sequential()
            model.add(Input(shape=(sequence_length, len(PoseAnalyzer.ANGLE_ORDER))))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer='adam', 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Train model
            log_dir = os.path.join('logs', f"{song_name}_{int(time.time())}")
            tensorboard_callback = TensorBoard(log_dir=log_dir)
            
            # Handle validation data based on sample size
            if len(X) == 1:
                # With only one sample, we can't do proper validation
                history = model.fit(X_train, y_train, 
                         epochs=50, 
                         verbose=1)
            else:
                history = model.fit(X_train, y_train, 
                         epochs=50, 
                         validation_data=(X_test, y_test),
                         callbacks=[tensorboard_callback],
                         verbose=1)
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
            model.save(model_path)
            
            # Save training history
            history_path = os.path.join(self.model_dir, f"{song_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(history.history, f)
                
            logger.info(f"Model trained and saved to {model_path}")
            return model_path, history.history
            
        except Exception as e:
            logger.error(f"Error training model for {song_name}: {e}")
            raise
    
    def load_model(self, song_name):
        model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        return None
    
    def model_exists(self, song_name):
        model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
        return os.path.exists(model_path)
    
    def pad_sequences(self, sequences, target_length):
        if not sequences:
            return np.zeros((0, target_length, len(PoseAnalyzer.ANGLE_ORDER)))
            
        # Find the maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        
        # Use the minimum of target_length and max_len
        actual_target = min(target_length, max_len)
        
        padded = np.zeros((len(sequences), actual_target, len(PoseAnalyzer.ANGLE_ORDER)))
        for i, seq in enumerate(sequences):
            if len(seq) > actual_target:
                # Truncate to target length
                padded[i] = seq[:actual_target]
            else:
                # Pad with zeros
                padded[i, :len(seq)] = seq
        return padded