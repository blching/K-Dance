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
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

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
    
class EnhancedDanceModelTrainer:
    def __init__(self, data_manager, use_enhanced_data=True):
        self.data_manager = data_manager
        self.model_dir = "dance_models"
        self.use_enhanced_data = use_enhanced_data
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define feature dimensions based on data type
        if self.use_enhanced_data:
            # Angles (8) + body_center (2) + body_scale (2) + body_orientation (1) + limb_lengths (4)
            self.feature_dim = 8 + 2 + 2 + 1 + 4
        else:
            # Only angles
            self.feature_dim = 8  # As defined in PoseAnalyzer.ANGLE_ORDER
    
    def prepare_data(self, song_name):
        """Prepare training data, handling both enhanced and regular data formats"""
        if self.use_enhanced_data:
            # Load enhanced reference (professional) performances
            X_ref = self.data_manager.load_enhanced_reference_data(song_name)
        else:
            # Load regular reference data
            X_ref = self.data_manager.load_reference_data(song_name)
        
        if not X_ref:
            logger.warning(f"No reference data found for {song_name}")
            return [], np.array([])
        
        # Create labels (perfect performance)
        y = [1] * len(X_ref)  # 1 = perfect performance
        
        # Add negative examples if available
        if self.use_enhanced_data:
            user_data = self.data_manager.load_enhanced_user_data(song_name)
        else:
            user_data = self.data_manager.load_user_data(song_name)
            
        if user_data:
            X_ref.extend(user_data)
            y.extend([0] * len(user_data))  # 0 = imperfect
        
        # Extract features based on data format
        X_features = []
        y_clean = []
        
        for i, seq in enumerate(X_ref):
            seq_features = []
            valid_sequence = True
            
            for frame in seq:
                if self.use_enhanced_data and isinstance(frame, dict):
                    # Enhanced data format
                    try:
                        frame_features = self.extract_features_from_enhanced_frame(frame)
                        seq_features.append(frame_features)
                    except Exception as e:
                        logger.warning(f"Error extracting features from enhanced frame: {e}")
                        valid_sequence = False
                        break
                elif not self.use_enhanced_data and isinstance(frame, list):
                    # Regular angle data format
                    if len(frame) == 8:  # Check if it has the expected 8 angles
                        seq_features.append(frame)
                    else:
                        logger.warning(f"Invalid frame length: {len(frame)} (expected 8)")
                        valid_sequence = False
                        break
                else:
                    logger.warning(f"Unexpected data format: {type(frame)}")
                    valid_sequence = False
                    break
            
            if valid_sequence and seq_features:
                X_features.append(seq_features)
                y_clean.append(y[i])
        
        logger.info(f"Prepared {len(X_features)} sequences with feature dimension {self.feature_dim}")
        return X_features, np.array(y_clean)
    
    def extract_features_from_enhanced_frame(self, frame):
        """Extract features from enhanced pose data frame"""
        features = []
        
        # Add angles (8 features)
        features.extend(frame['angles'])
        
        # Add body center (2 features)
        features.extend(frame['body_center'])
        
        # Add body scale (2 features)
        features.extend(frame['body_scale'])
        
        # Add body orientation (1 feature)
        features.append(frame['body_orientation'])
        
        # Add limb lengths (4 features)
        limb_lengths = frame['limb_lengths']
        features.extend([
            limb_lengths.get('left_arm', 0),
            limb_lengths.get('right_arm', 0),
            limb_lengths.get('left_leg', 0),
            limb_lengths.get('right_leg', 0)
        ])
        
        # Optional: Add selected raw positions if needed
        # raw_positions = frame['raw_positions']
        # features.extend(raw_positions.get('LEFT_HIP', [0, 0]))
        # features.extend(raw_positions.get('RIGHT_HIP', [0, 0]))
        
        return features
    
    def train_model(self, song_name, sequence_length=30):
        """Train a model for the given song"""
        try:
            X, y = self.prepare_data(song_name)
            
            if not X or len(X) == 0:
                raise ValueError(f"No valid training data found for {song_name}")
            
            logger.info(f"Training model for {song_name} with {len(X)} sequences")
            
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
            
            # Build LSTM model
            model = self.build_model(sequence_length)
            
            # Set up callbacks
            callbacks = self.setup_callbacks(song_name)
            
            # Train model
            if len(X) == 1:
                # With only one sample, we can't do proper validation
                history = model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    verbose=1,
                    callbacks=callbacks
                )
            else:
                history = model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
            model.save(model_path)
            
            # Save training history
            history_path = os.path.join(self.model_dir, f"{song_name}_history.json")
            with open(history_path, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
                json.dump(history_dict, f)
                
            logger.info(f"Model trained and saved to {model_path}")
            return model_path, history.history
            
        except Exception as e:
            logger.error(f"Error training model for {song_name}: {e}")
            raise
    
    def build_model(self, sequence_length):
        """Build the LSTM model architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(sequence_length, self.feature_dim)))
        
        # First LSTM layer
        model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Third LSTM layer
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Built model with input shape: ({sequence_length}, {self.feature_dim})")
        return model
    
    def setup_callbacks(self, song_name):
        """Set up training callbacks"""
        log_dir = os.path.join('logs', f"{song_name}_{int(time.time())}")
        
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=1),
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        return callbacks
    
    def pad_sequences(self, sequences, target_length):
        """Pad sequences to a uniform length"""
        if not sequences:
            return np.zeros((0, target_length, self.feature_dim))
            
        # Find the maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        
        # Use the minimum of target_length and max_len
        actual_target = min(target_length, max_len)
        
        padded = np.zeros((len(sequences), actual_target, self.feature_dim))
        for i, seq in enumerate(sequences):
            if len(seq) > actual_target:
                # Truncate to target length
                padded[i] = seq[:actual_target]
            else:
                # Pad with zeros
                padded[i, :len(seq)] = seq
        
        logger.info(f"Padded sequences to shape: {padded.shape}")
        return padded
    
    def load_model(self, song_name):
        """Load a trained model for the given song"""
        model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        return None
    
    def model_exists(self, song_name):
        """Check if a model exists for the given song"""
        model_path = os.path.join(self.model_dir, f"{song_name}_model.h5")
        return os.path.exists(model_path)
    
    def evaluate_model(self, song_name, X_test=None, y_test=None):
        """Evaluate the trained model"""
        model = self.load_model(song_name)
        if not model:
            logger.error(f"No model found for {song_name}")
            return None
        
        if X_test is None or y_test is None:
            # Prepare test data
            X, y = self.prepare_data(song_name)
            if not X:
                return None
                
            # Split data
            if len(X) > 1:
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_test = self.pad_sequences(X_test, 30)  # Use default sequence length
            else:
                logger.warning("Insufficient data for proper evaluation")
                return None
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        
        # Create evaluation report
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        report = {metric: value for metric, value in zip(metrics, evaluation)}
        
        logger.info(f"Model evaluation for {song_name}: {report}")
        return report
    
    def predict(self, song_name, sequence):
        """Make a prediction using the trained model"""
        model = self.load_model(song_name)
        if not model:
            logger.error(f"No model found for {song_name}")
            return None
        
        # Prepare the sequence for prediction
        if self.use_enhanced_data and isinstance(sequence[0], dict):
            # Extract features from enhanced data
            prepared_sequence = []
            for frame in sequence:
                prepared_sequence.append(self.extract_features_from_enhanced_frame(frame))
            sequence = prepared_sequence
        
        # Pad the sequence
        padded_sequence = self.pad_sequences([sequence], 30)  # Use default sequence length
        
        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)
        return prediction[0][0]  # Return scalar value
