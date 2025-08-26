import os
import cv2

from AudioSynchronizer import AudioSynchronizer
from PoseProcessing import PoseAnalyzer, EnhancedPoseAnalyzer

import cv2
import numpy as np
from PoseProcessing import EnhancedPoseAnalyzer


class VideoTrainer:
    def __init__(self, pose_detector, data_manager):
        self.pose_detector = pose_detector
        self.data_manager = data_manager
        self.audio_synchronizer = AudioSynchronizer()
        self.enhanced_pose_analyzer = EnhancedPoseAnalyzer()
    
    def process_video(self, video_path, song_name, performer_type="pro", show_preview=False, enhanced=True):
        """
        Process a video file to extract pose data for training
        """
        if enhanced:
            return self.process_video_enhanced(video_path, song_name, performer_type, show_preview)
        else:
            # Original implementation
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return False
        
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Processing video: {video_path}")
            print(f"FPS: {fps}, Total frames: {total_frames}")
            
            # Extract audio from the video for later use
            print("Extracting audio from video...")
            audio_data, sample_rate = self.audio_synchronizer.extract_audio_from_video(video_path)
            
            frame_data = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.pose_detector.process_frame(frame)
                
                if results.pose_landmarks:
                    # Extract angles
                    landmarks = results.pose_landmarks.landmark
                    angles = PoseAnalyzer.extract_angles_as_list(landmarks, self.pose_detector.mp_pose)
                    frame_data.append(angles)
                    
                    # Draw landmarks if preview is enabled
                    if show_preview:
                        frame = cv2.resize(frame, (350, 600))
                        self.pose_detector.draw_landmarks(frame, results.pose_landmarks)
                        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Video Processing Preview', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Save the extracted data
            if frame_data:
                save_path = self.data_manager.save_session(frame_data, song_name, performer_type)
                print(f"Saved {len(frame_data)} frames to {save_path}")
                
                # Save the audio data if extracted successfully
                if audio_data is not None:
                    # Create a directory for audio files if it doesn't exist
                    audio_dir = os.path.join(self.data_manager.data_dir, "audio")
                    os.makedirs(audio_dir, exist_ok=True)
                    
                    # Save audio as numpy array
                    audio_path = os.path.join(audio_dir, f"{song_name}_{performer_type}.npy")
                    np.save(audio_path, {
                        'audio_data': audio_data,
                        'sample_rate': sample_rate,
                        'video_path': video_path
                    })
                    print(f"Audio data saved to {audio_path}")
                
                return True
            else:
                print("No pose data extracted from video")
                return False 
    
    def batch_process_videos(self, video_dir, song_name, performer_type="pro", show_preview=False):
        """
        Process all videos in a directory for training
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        processed_count = 0
        
        for file in os.listdir(video_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(video_dir, file)
                if self.process_video(video_path, song_name, performer_type, show_preview):
                    processed_count += 1
        
        print(f"Processed {processed_count} videos for {song_name}")
        return processed_count
    
    def process_video_enhanced(self, video_path, song_name, performer_type="pro", show_preview=False):
        """
        Process a video file to extract enhanced pose data for training
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video with enhanced data: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.pose_detector.process_frame(frame)
            
            if results.pose_landmarks:
                # Extract enhanced pose data
                landmarks = results.pose_landmarks.landmark
                pose_data = self.enhanced_pose_analyzer.extract_complete_pose_data(landmarks, self.pose_detector.mp_pose)
                frame_data.append(pose_data)
                
                # Draw landmarks if preview is enabled
                if show_preview:
                    self.pose_detector.draw_landmarks(frame, results.pose_landmarks)
                    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Enhanced Video Processing Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Save the extracted data
        if frame_data:
            save_path = self.data_manager.save_enhanced_session(frame_data, song_name, performer_type)
            print(f"Saved {len(frame_data)} enhanced frames to {save_path}")
            return True
        else:
            print("No pose data extracted from video")
            return False
