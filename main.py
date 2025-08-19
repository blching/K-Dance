import os
import time
import mediapipe as mp
import cv2
import numpy as np
import json

class PoseAnalyzer:
    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180:
            angle = 360 - angle

        return angle

    @staticmethod
    def normalize_landmarks(landmarks, mp_pose):
        l_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        l_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        r_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

        pelvis = (l_hip + r_hip) / 2.0
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        shoulder_width = max(shoulder_width, 1e-5)  # Avoid division by zero

        normalized = []
        for lm in landmarks:
            x, y = lm.x, lm.y
            normed = [(x - pelvis[0]) / shoulder_width,
                      (y - pelvis[1]) / shoulder_width]
            normalized.append(normed)
        return normalized

    @staticmethod
    def extract_angles(norm, mp_pose):
        angles = {}
        angles["left_elbow"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            norm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        angles["right_elbow"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            norm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        angles["left_shoulder"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.LEFT_HIP.value],
            norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        )
        angles["right_shoulder"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.RIGHT_HIP.value],
            norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        )
        angles["left_knee"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.LEFT_HIP.value],
            norm[mp_pose.PoseLandmark.LEFT_KNEE.value],
            norm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        angles["right_knee"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.RIGHT_HIP.value],
            norm[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            norm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        angles["left_hip"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.LEFT_HIP.value],
            norm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        angles["right_hip"] = PoseAnalyzer.calculate_angle(
            norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            norm[mp_pose.PoseLandmark.RIGHT_HIP.value],
            norm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        return angles


class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)
    
    def draw_landmarks(self, frame, landmarks):
        self.mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS
        )

# Data Management
class DanceDataManager:
    def __init__(self, data_dir="dance_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_session(self, session_data, song_name, performer_type="user"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{song_name}_{performer_type}_{timestamp}.json"
        path = os.path.join(self.data_dir, filename)
        
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)
        return path
    
    def load_reference_data(self, song_name):
        """Load professional performance data for a song"""
        ref_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "pro" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    ref_data.append(json.load(f))
        return ref_data
    
    def load_user_data(self, song_name):
        """Load user performance data for a song"""
        user_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "user" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    user_data.append(json.load(f))
        return user_data

class VideoProcessor:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.session = []
        self.pose_detector = PoseDetector()
    
    def process_video(self, output_file="session.json"):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            try:
                frame = self._process_frame(frame)
                cv2.imshow('Output', frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                break
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        self._save_session(output_file)
        self._cleanup()
    
    def _process_frame(self, frame):
        frame = cv2.resize(frame, (350, 600))
        results = self.pose_detector.process_frame(frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self._store_landmarks(landmarks)
            self.pose_detector.draw_landmarks(frame, results.pose_landmarks)
        return frame
    
    def _store_landmarks(self, landmarks):
        frame_data = [{
            "x": lm.x, 
            "y": lm.y, 
            "z": lm.z, 
            "vis": lm.visibility
        } for lm in landmarks]
        self.session.append(frame_data)
    
    def _save_session(self, output_file):
        with open(output_file, "w") as f:
            json.dump(self.session, f, indent=2)
    
    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # For video file
    processor = VideoProcessor('test_winterDirtyWork.mp4')
    
    # For webcam
    # processor = VideoProcessor(0)
    
    # Output file
    processor.process_video("test_session.json")