from collections import deque
import numpy as np
from PoseProcessing import PoseAnalyzer

class DanceEvaluator:
    def __init__(self, model, reference_sequence, audio_synchronizer=None, enhanced=False):
        self.model = model
        self.reference_sequence = reference_sequence
        self.audio_synchronizer = audio_synchronizer
        self.enhanced = enhanced
        self.sequence_length = 30
        self.current_sequence = deque(maxlen=self.sequence_length)
        self.beat_sequences = {}  # Store sequences for each beat
        self.current_beat = 0
    
    def add_frame(self, frame_data, beat_idx=None):
        """Add current frame data to evaluation sequence"""
        if beat_idx is not None and self.audio_synchronizer:
            # Store frame by beat
            if beat_idx not in self.beat_sequences:
                self.beat_sequences[beat_idx] = deque(maxlen=5)  # Store last 5 frames for this beat
            self.beat_sequences[beat_idx].append(frame_data)
            self.current_beat = beat_idx
        
        # Also add to the main sequence
        self.current_sequence.append(frame_data)
    
    def evaluate(self):
        """Evaluate current sequence against model"""
        if len(self.current_sequence) < 5:  # Minimum frames to evaluate
            return None
        
        # Prepare sequence
        if self.enhanced:
            # For enhanced data, we might need to extract specific features
            sequence_features = []
            for frame in self.current_sequence:
                if isinstance(frame, dict):  # Enhanced data
                    # Extract the features used by the model
                    features = frame['angles'].copy()
                    features.extend(frame['body_center'])
                    features.extend(frame['body_scale'])
                    # Add other features as needed by the model
                    sequence_features.append(features)
                else:  # Angle-only data
                    sequence_features.append(frame)
            sequence = np.array([sequence_features])
        else:
            sequence = np.array([list(self.current_sequence)])
        
        sequence = self.pad_sequence(sequence)
        
        # Get model prediction
        score = self.model.predict(sequence, verbose=0)[0][0]
        
        # Compare to reference
        ref_score = self.compare_to_reference()
        
        # Generate beat-specific feedback if audio is synchronized
        feedback = self.generate_feedback(score, ref_score)
        
        return {
            "performance_score": float(score),
            "reference_similarity": float(ref_score),
            "current_beat": self.current_beat if self.audio_synchronizer else None,
            "on_beat": self.audio_synchronizer.is_on_beat() if self.audio_synchronizer else None,
            "feedback": feedback
        }
    
    def pad_sequence(self, sequence):
        padded = np.zeros((1, self.sequence_length, len(PoseAnalyzer.ANGLE_ORDER)))
        seq_data = sequence[0]
        if len(seq_data) > self.sequence_length:
            padded[0] = seq_data[-self.sequence_length:]
        else:
            padded[0, :len(seq_data)] = seq_data
        return padded
    
    def compare_to_reference(self):
        """Calculate similarity to reference sequence"""
        if len(self.current_sequence) < 10:
            return 0
        
        current = np.array(list(self.current_sequence))
        
        # Find the best matching segment in the reference
        best_similarity = 0
        ref_len = len(self.reference)
        current_len = len(current)
        
        # Slide window through reference to find best match
        for i in range(0, ref_len - current_len + 1, 5):  # Step by 5 frames
            ref_segment = np.array(self.reference[i:i+current_len])
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(current - ref_segment))
            similarity = max(0, 1 - mae/50)  # Convert to similarity score
            
            if similarity > best_similarity:
                best_similarity = similarity
        
        return best_similarity
    
    def generate_feedback(self, score, similarity):
        """Generate actionable feedback based on scores"""
        feedback = []
        
        if score < 0.7:
            feedback.append("Overall timing and rhythm need improvement")
        
        if similarity < 0.6:
            feedback.append("Focus on matching the choreography's movements")
        
        # Beat-specific feedback (if we have enough frames)
        if self.audio_synchronizer and self.current_beat > 0:
            # Check if we're on beat
            if not self.audio_synchronizer.is_on_beat():
                feedback.append("Try to synchronize your movements with the beat")
            
            # Compare to reference for this specific beat
            if self.current_beat < len(self.reference):
                current_frame = list(self.current_sequence)[-1] if self.current_sequence else None
                ref_frame = self.reference[self.current_beat]
                
                if current_frame is not None:
                    for i, joint in enumerate(PoseAnalyzer.ANGLE_ORDER):
                        diff = abs(current_frame[i] - ref_frame[i])
                        if diff > 30:
                            feedback.append(f"Adjust your {joint.replace('_', ' ')} angle on this beat")
        
        return feedback if feedback else ["Great job! Keep going!"]