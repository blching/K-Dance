import os
import tempfile
import threading
import time
import librosa
import sounddevice as sd
import soundfile as sf
from collections import deque
from moviepy import VideoFileClip

class AudioSynchronizer:
    def __init__(self):
        self.current_audio = None
        self.sample_rate = None
        self.beat_times = None
        self.current_beat_idx = 0
        self.audio_start_time = 0
        self.chorus_start = 0
        self.chorus_end = 0
        self.is_playing = False
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from a video file and return the audio data"""
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Updated write_audiofile call without verbose parameter
            audio.write_audiofile(temp_path, logger=None)
            
            # Load the audio with librosa
            audio_data, sr = librosa.load(temp_path, sr=None)
            
            # Clean up
            os.unlink(temp_path)
            video.close()
            
            return audio_data, sr
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None, None
    
    def load_audio_from_video(self, video_path, chorus_start=0, chorus_end=30):
        """Load audio from a video file and extract beat information"""
        try:
            # Extract audio from video
            self.current_audio, self.sample_rate = self.extract_audio_from_video(video_path)
            
            if self.current_audio is None:
                return False
            
            # Set chorus segment
            self.chorus_start = chorus_start
            audio_duration = len(self.current_audio) / self.sample_rate
            self.chorus_end = min(chorus_end, audio_duration)
            
            # Extract beats from the chorus segment
            start_sample = int(self.chorus_start * self.sample_rate)
            end_sample = int(self.chorus_end * self.sample_rate)
            chorus_audio = self.current_audio[start_sample:end_sample]
            
            tempo, beat_frames = librosa.beat.beat_track(y=chorus_audio, sr=self.sample_rate)
            self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            
            print(f"Loaded audio with {len(self.beat_times)} beats at {tempo:.2f} BPM")
            print(f"Chorus segment: {chorus_start}s to {self.chorus_end}s")
            return True
        except Exception as e:
            print(f"Error loading audio from video: {e}")
            return False
    
    def load_audio_directly(self, audio_path, chorus_start=0, chorus_end=30):
        """Load audio directly from an audio file"""
        try:
            # Load audio file
            self.current_audio, self.sample_rate = librosa.load(audio_path, sr=None)
            
            # Set chorus segment
            self.chorus_start = chorus_start
            audio_duration = len(self.current_audio) / self.sample_rate
            self.chorus_end = min(chorus_end, audio_duration)
            
            # Extract beats from the chorus segment
            start_sample = int(self.chorus_start * self.sample_rate)
            end_sample = int(self.chorus_end * self.sample_rate)
            chorus_audio = self.current_audio[start_sample:end_sample]
            
            tempo, beat_frames = librosa.beat.beat_track(y=chorus_audio, sr=self.sample_rate)
            self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            
            print(f"Loaded audio with {len(self.beat_times)} beats at {tempo:.2f} BPM")
            print(f"Chorus segment: {chorus_start}s to {self.chorus_end}s")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def play_chorus(self):
        """Play only the chorus segment of the audio"""
        if self.current_audio is not None:
            # Extract chorus segment
            start_sample = int(self.chorus_start * self.sample_rate)
            end_sample = int(self.chorus_end * self.sample_rate)
            chorus_segment = self.current_audio[start_sample:end_sample]
            
            # Play the segment
            self.audio_start_time = time.time()
            sd.play(chorus_segment, self.sample_rate)
            self.is_playing = True
            
            # Set a timer to stop playback after chorus duration
            chorus_duration = self.chorus_end - self.chorus_start
            threading.Timer(chorus_duration, self.stop_audio).start()
            
            return True
        return False
    
    def stop_audio(self):
        """Stop audio playback"""
        sd.stop()
        self.is_playing = False
    
    def get_current_beat(self):
        """Get the current beat based on playback time"""
        if self.beat_times is None or self.audio_start_time == 0:
            return None
        
        current_time = time.time() - self.audio_start_time
        # Adjust for chorus start
        current_time += self.chorus_start
        
        # Find the closest beat
        for i, beat_time in enumerate(self.beat_times):
            if beat_time > current_time:
                return i - 1 if i > 0 else 0
        
        return len(self.beat_times) - 1
    
    def is_on_beat(self, tolerance=0.1):
        """Check if we're currently on a beat"""
        if self.beat_times is None or self.audio_start_time == 0:
            return False
        
        current_time = time.time() - self.audio_start_time
        # Adjust for chorus start
        current_time += self.chorus_start
        
        # Check if we're close to any beat
        for beat_time in self.beat_times:
            if abs(current_time - beat_time) < tolerance:
                return True
        
        return False