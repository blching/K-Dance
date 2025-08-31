import threading
import time
import cv2
import os

from AudioSynchronizer import AudioSynchronizer
from DanceDataManager import DanceDataManager
from DanceEvaluator import DanceEvaluator
from DanceFeedback import DanceFeedbackSystem
from DanceModelTrainer import EnhancedDanceModelTrainer
from DanceMoveVisualizer import EnhancedDanceMoveVisualizer
from PoseProcessing import PoseAnalyzer, PoseDetector, EnhancedPoseAnalyzer
from VideoTrainer import VideoTrainer


class DanceCoachApp:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.data_manager = DanceDataManager()
        self.model_trainer = EnhancedDanceModelTrainer(self.data_manager)
        self.video_trainer = VideoTrainer(self.pose_detector, self.data_manager)
        self.audio_synchronizer = AudioSynchronizer()
        self.enhanced_pose_analyzer = EnhancedPoseAnalyzer()
        self.enhanced_visualizer = EnhancedDanceMoveVisualizer()
        self.current_song = None
        self.evaluator = None
        self.feedback_system = None
        self.cap = None
        self.session_active = False
        self.use_enhanced_data = True  # Flag to control enhanced data usage
    
    def show_preview(self, duration=5):
        """Show a preview screen for users to position themselves"""
        print("Get ready! Positioning preview starting...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (350, 600))
            
            # Add instructions
            cv2.putText(frame, "Position yourself in frame", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Starting in: {int(duration - (time.time() - start_time))}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw a frame border
            cv2.rectangle(frame, (50, 100), (300, 550), (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('K-Pop Dance Coach - Preview', frame)
            
            # Check for quit
            if cv2.waitKey(1) == ord('q'):
                return False
        
        return True
    
    def countdown(self, seconds=5):
        """Show a countdown before starting"""
        for i in range(seconds, 0, -1):
            ret, frame = self.cap.read()
            if not ret:
                return False
            
            frame = cv2.resize(frame, (350, 600))
            cv2.putText(frame, f"Starting in: {i}", (100, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('K-Pop Dance Coach', frame)
            cv2.waitKey(1000)  # Wait 1 second
        
        return True
    
    def select_song(self, song_name, audio_source=None, chorus_start=0, chorus_end=30):
        """Select a song and load its model with chorus timing"""
        self.current_song = song_name
        
        # Load or train model
        model = self.model_trainer.load_model(song_name)
        if not model:
            print(f"Training model for {song_name}...")
            try:
                model_path, history = self.model_trainer.train_model(song_name)
                model = self.model_trainer.load_model(song_name)
                print(f"Model trained and saved to {model_path}")
            except Exception as e:
                print(f"Error training model: {e}")
                return False
        
        # Load reference sequence
        ref_data = self.data_manager.load_reference_data(song_name)
        if not ref_data:
            print(f"No reference data found for {song_name}")
            return False
        
        # Load audio if provided
        if audio_source:
            if isinstance(audio_source, str):
                if audio_source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # It's a video file, extract audio from it
                    success = self.audio_synchronizer.load_audio_from_video(audio_source, chorus_start, chorus_end)
                else:
                    # It's an audio file
                    success = self.audio_synchronizer.load_audio_directly(audio_source, chorus_start, chorus_end)
                
                if not success:
                    print("Failed to load audio. Continuing without audio synchronization.")
            else:
                print("Invalid audio source type. Continuing without audio synchronization.")
        
        # Use the first reference sequence
        ref_sequence = ref_data[0]
        
        # Create evaluator with audio synchronizer
        self.evaluator = DanceEvaluator(model, ref_sequence, self.audio_synchronizer)
        self.feedback_system = DanceFeedbackSystem(self.pose_detector, self.evaluator)
        return True
    
    def start_webcam_session(self, chorus_start=0, chorus_end=30, enhanced=True):
        """Start a webcam dance session with enhanced data option"""
        if enhanced is None:
            enhanced = self.use_enhanced_data
        
        if not self.current_song:
            raise RuntimeError("No song selected")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Show preview
        if not self.show_preview(5):
            self.cap.release()
            cv2.destroyAllWindows()
            return
        
        # Show countdown
        if not self.countdown(3):
            self.cap.release()
            cv2.destroyAllWindows()
            return
        
        # Start audio playback
        if self.audio_synchronizer.current_audio is not None:
            self.audio_synchronizer.play_chorus()
        
        session_data = []
        self.session_active = True
        
        # Get session start time
        session_start = time.time()
        chorus_duration = chorus_end - chorus_start
        
        while self.cap.isOpened() and self.session_active:
            # Check if chorus duration has elapsed
            if time.time() - session_start > chorus_duration:
                break
            
            _, frame = self.cap.read()
            if frame is None:
                break
            
            try:
                # Process frame
                frame = cv2.resize(frame, (350, 600))
                results = self.pose_detector.process_frame(frame)
                
                if results.pose_landmarks:
                    # Extract and store data
                    landmarks = results.pose_landmarks.landmark
                    angles = EnhancedPoseAnalyzer.extract_angles_as_list(landmarks, self.pose_detector.mp_pose)

                    # Get current beat if audio is synchronized
                    beat_idx = None
                    if self.audio_synchronizer:
                        beat_idx = self.audio_synchronizer.get_current_beat()

                    if enhanced:
                        pose_data = self.enhanced_pose_analyzer.extract_complete_pose_data(landmarks, self.pose_detector.mp_pose)
                        session_data.append(pose_data)
                        # For evaluation, we might need to extract just the angles or use the full pose data
                        evaluation_data = pose_data['angles']  # Or use the full pose_data
                        self.evaluator.add_frame(evaluation_data, beat_idx) # Add to evaluator with beat information
                    else:
                        angles = EnhancedPoseAnalyzer.extract_angles_as_list(landmarks, self.pose_detector.mp_pose)
                        session_data.append(angles)
                        evaluation_data = angles
                        self.evaluator.add_frame(angles, beat_idx) # Add to evaluator with beat information
                    
                    # Evaluate performance
                    evaluation = self.evaluator.evaluate()
                    
                    # Draw landmarks
                    self.pose_detector.draw_landmarks(frame, results.pose_landmarks)
                    
                    # Display feedback
                    if evaluation:
                        frame = self.feedback_system.visualize_feedback(frame, evaluation)
                    
                    # Display time remaining
                    time_left = max(0, chorus_duration - (time.time() - session_start))
                    cv2.putText(frame, f"Time: {time_left:.1f}s", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('K-Pop Dance Coach', frame)
                
            except Exception as e:
                print(f"Error: {e}")
                break
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Stop audio playback
        if self.audio_synchronizer:
            self.audio_synchronizer.stop_audio()
        
        # Save session data
        if session_data:
            self.data_manager.save_session(session_data, self.current_song, "user")
            print(f"Session data saved for {self.current_song}")
        
        # Show session complete message
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (350, 600))
            cv2.putText(frame, "Session Complete!", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('K-Pop Dance Coach', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.session_active = False
    
    def start_video_session(self, video_path, chorus_start=0, chorus_end=30, enhanced=None):
        """Start a dance session using a video file"""
        if not self.current_song:
            raise RuntimeError("No song selected")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return
        
        # Load audio from the video if no audio is already loaded
        if self.audio_synchronizer.current_audio is None:
            print("Extracting audio from video for synchronization...")
            self.audio_synchronizer.load_audio_from_video(video_path, chorus_start, chorus_end)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        # Adjust chorus end if needed
        chorus_end = min(chorus_end, video_duration)
        chorus_duration = chorus_end - chorus_start
        
        print(f"Video duration: {video_duration:.1f}s")
        print(f"Analyzing chorus segment: {chorus_start}s to {chorus_end}s")
        
        # Seek to chorus start
        start_frame = int(chorus_start * fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        session_data = []
        self.session_active = True
        
        # Start audio playback
        if self.audio_synchronizer.current_audio is not None:
            self.audio_synchronizer.play_chorus()
        
        # Process video frames
        frame_count = start_frame
        end_frame = int(chorus_end * fps)
        
        while self.cap.isOpened() and frame_count <= end_frame and self.session_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            try:
                # Process frame
                frame = cv2.resize(frame, (350, 600))
                results = self.pose_detector.process_frame(frame)
                
                if results.pose_landmarks:
                    # Extract and store data
                    landmarks = results.pose_landmarks.landmark
                    angles = EnhancedPoseAnalyzer.extract_angles_as_list(landmarks, self.pose_detector.mp_pose)

                    # Get current beat if audio is synchronized
                    beat_idx = None
                    if self.audio_synchronizer:
                        beat_idx = self.audio_synchronizer.get_current_beat()

                    if enhanced:
                        pose_data = self.enhanced_pose_analyzer.extract_complete_pose_data(landmarks, self.pose_detector.mp_pose)
                        session_data.append(pose_data)
                        # For evaluation, we might need to extract just the angles or use the full pose data
                        evaluation_data = pose_data['angles']  # Or use the full pose_data
                        self.evaluator.add_frame(evaluation_data, beat_idx) # Add to evaluator with beat information
                    else:
                        angles = EnhancedPoseAnalyzer.extract_angles_as_list(landmarks, self.pose_detector.mp_pose)
                        session_data.append(angles)
                        evaluation_data = angles
                        self.evaluator.add_frame(angles, beat_idx) # Add to evaluator with beat information
                    
                    # Evaluate performance
                    evaluation = self.evaluator.evaluate()
                    
                    # Draw landmarks
                    self.pose_detector.draw_landmarks(frame, results.pose_landmarks)
                    
                    # Display feedback
                    if evaluation:
                        frame = self.feedback_system.visualize_feedback(frame, evaluation)
                    
                    # Display current time
                    current_time = frame_count / fps
                    cv2.putText(frame, f"Time: {current_time:.1f}s", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('K-Pop Dance Coach', frame)
                
                # Calculate delay based on video FPS
                delay = int(1000 / fps)
                if cv2.waitKey(delay) == ord('q'):
                    break
                
                frame_count += 1
                
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # Stop audio playback
        if self.audio_synchronizer:
            self.audio_synchronizer.stop_audio()
        
        # Save session data
        if session_data:
            self.data_manager.save_session(session_data, self.current_song, "user")
            print(f"Session data saved for {self.current_song}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.session_active = False
    
    def add_professional_video(self, video_path, song_name, show_preview=False, enhanced=True):
        """
        Add a professional video to train the system
        """
        print(f"Adding professional video for {song_name}: {video_path}")
        success = self.video_trainer.process_video(video_path, song_name, "pro", show_preview, enhanced)
        if success:
            print(f"Professional video added successfully for {song_name}")
            # Retrain the model with the new data
            if self.model_trainer.model_exists(song_name):
                print("Retraining model with new data...")
                try:
                    self.model_trainer.train_model(song_name)
                except Exception as e:
                    print(f"Error retraining model: {e}")
        return success
    
    def batch_add_professional_videos(self, video_dir, song_name, show_preview=False, enhanced=True):
        """
        Add all professional videos in a directory
        """
        print(f"Batch adding professional videos from {video_dir} for {song_name}")
        count = 0
        for file in os.listdir(video_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(video_dir, file)
                if self.video_trainer.process_video(video_path, song_name, "pro", show_preview, enhanced):
                    count += 1
        
        print(f"Processed {count} videos for {song_name}")
        if count > 0 and self.model_trainer.model_exists(song_name):
            print("Retraining model with new data...")
            self.model_trainer.train_model(song_name)
        return count
    
    def get_available_songs(self):
        """Get list of all available songs"""
        return self.data_manager.get_all_songs()
    
    def get_song_status(self, song_name):
        """Get training status for a song"""
        has_reference = len(self.data_manager.load_reference_data(song_name)) > 0
        has_model = self.model_trainer.model_exists(song_name)
        
        return {
            "song_name": song_name,
            "has_reference_data": has_reference,
            "has_trained_model": has_model,
            "reference_count": len(self.data_manager.load_reference_data(song_name)),
            "user_count": len(self.data_manager.load_user_data(song_name))
        }
    
    def has_sufficient_data(self, song_name, min_samples=2):
        """Check if we have enough data to train a model"""
        ref_data = self.data_manager.load_reference_data(song_name)
        return len(ref_data) >= min_samples
    
    def visualize_song_dance(self, song_name, sequence_index=0, output_gif=None, output_sheet=None):
        """Visualize the dance moves for a specific song"""
        visualizer = EnhancedDanceMoveVisualizer(self.data_manager)
        
        # Create animation
        ani = visualizer.visualize_dance_moves(song_name, sequence_index, output_gif)
        
        # Create dance move sheet
        if output_sheet:
            fig = visualizer.create_dance_move_sheet(song_name, output_sheet)
        
        return ani
    
    def visualize_enhanced_dance(self, song_name, sequence_index=0, 
                               output_gif=None, output_sheet=None, 
                               output_spatial=None):
        """
        Visualize dance moves using enhanced data with multiple output options
        
        Args:
            song_name: Name of the song to visualize
            sequence_index: Index of the sequence to visualize (default: 0)
            output_gif: Path to save animated GIF (optional)
            output_sheet: Path to save pose sheet (optional)
            output_spatial: Path to save spatial movement visualization (optional)
        
        Returns:
            animation object if output_gif is None, otherwise None
        """
        try:
            # Load enhanced data
            if self.use_enhanced_data:
                enhanced_data = self.data_manager.load_enhanced_reference_data(song_name)
            else:
                # Fall back to regular data if enhanced is not available
                regular_data = self.data_manager.load_reference_data(song_name)
                # Convert regular data to enhanced format if needed
                enhanced_data = self.convert_to_enhanced_format(regular_data)
            
            if not enhanced_data:
                print(f"No enhanced data found for {song_name}")
                return None
            
            if sequence_index >= len(enhanced_data):
                print(f"Sequence index {sequence_index} out of range. Available sequences: {len(enhanced_data)}")
                return None
            
            sequence = enhanced_data[sequence_index]
            
            # Create output directory if it doesn't exist
            if output_gif:
                os.makedirs(os.path.dirname(output_gif), exist_ok=True)
            if output_sheet:
                os.makedirs(os.path.dirname(output_sheet), exist_ok=True)
            if output_spatial:
                os.makedirs(os.path.dirname(output_spatial), exist_ok=True)
            
            # Create visualizations
            results = {}
            
            # Animated GIF
            if output_gif:
                print(f"Creating animated GIF: {output_gif}")
                animation = self.enhanced_visualizer.visualize_enhanced_dance_moves(
                    sequence, output_gif
                )
                results['animation'] = animation
                print(f"GIF saved to {output_gif}")
            
            # Pose sheet
            if output_sheet:
                print(f"Creating pose sheet: {output_sheet}")
                fig = self.enhanced_visualizer.create_enhanced_dance_move_sheet(
                    sequence, output_sheet
                )
                results['pose_sheet'] = fig
                print(f"Pose sheet saved to {output_sheet}")
            
            # Spatial movement visualization
            if output_spatial:
                print(f"Creating spatial movement visualization: {output_spatial}")
                plt_obj = self.enhanced_visualizer.visualize_spatial_movement(
                    sequence, output_spatial
                )
                results['spatial_visualization'] = plt_obj
                print(f"Spatial visualization saved to {output_spatial}")
            
            return results
            
        except Exception as e:
            print(f"Error visualizing enhanced dance: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    app = DanceCoachApp()
    
    # Get available songs
    songs = app.get_available_songs()
    print("Available songs:", songs)
    
    if songs:
        # Visualize the first song
        song_name = "This is For"
        print(f"Visualizing dance for: {song_name}")
        
        # Create visualization
        animation = app.visualize_enhanced_dance(
            song_name, 
            output_gif=f"visualizations/{song_name}_dance.gif",
            output_sheet=f"visualizations/{song_name}_poses.png"
        )
        