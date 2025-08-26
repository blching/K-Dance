import cv2
import numpy as np
import mediapipe as mp

class PoseAnalyzer:
    ANGLE_ORDER = [
        "left_elbow", "right_elbow", 
        "left_shoulder", "right_shoulder",
        "left_knee", "right_knee",
        "left_hip", "right_hip"
    ]
    
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
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        pelvis = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        shoulder_width = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        
        # Normalize all points relative to pelvis and shoulder width
        normalized = []
        for lm in landmarks:
            x, y = lm.x, lm.y
            normed = [(x - pelvis[0]) / shoulder_width,
                    (y - pelvis[1]) / shoulder_width]
            normalized.append(normed)
        return normalized

    @staticmethod
    def extract_angles_as_list(landmarks, mp_pose):
        normalized = PoseAnalyzer.normalize_landmarks(landmarks, mp_pose)
        angles = {}
        
        # Arms
        angles["left_elbow"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            normalized[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        angles["right_elbow"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            normalized[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        angles["left_shoulder"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.LEFT_HIP.value],
            normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        )
        angles["right_shoulder"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.RIGHT_HIP.value],
            normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        )

        # Legs
        angles["left_knee"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.LEFT_HIP.value],
            normalized[mp_pose.PoseLandmark.LEFT_KNEE.value],
            normalized[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        angles["right_knee"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.RIGHT_HIP.value],
            normalized[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            normalized[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        angles["left_hip"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.LEFT_HIP.value],
            normalized[mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        angles["right_hip"] = PoseAnalyzer.calculate_angle(
            normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            normalized[mp_pose.PoseLandmark.RIGHT_HIP.value],
            normalized[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        return [angles[key] for key in PoseAnalyzer.ANGLE_ORDER]


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
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
        
class EnhancedPoseAnalyzer(PoseAnalyzer):
    @staticmethod
    def extract_complete_pose_data(landmarks, mp_pose):
        """Extract both angles and positional data"""
        # Get angles (existing functionality)
        angles = PoseAnalyzer.extract_angles_as_list(landmarks, mp_pose)
        
        # Get normalized landmarks (existing functionality)
        normalized_landmarks = PoseAnalyzer.normalize_landmarks(landmarks, mp_pose)
        
        # Extract additional positional information
        pose_data = {
            'angles': angles,
            'normalized_landmarks': normalized_landmarks,
            'body_center': EnhancedPoseAnalyzer.calculate_body_center(landmarks, mp_pose),
            'body_scale': EnhancedPoseAnalyzer.calculate_body_scale(landmarks, mp_pose),
            'body_orientation': EnhancedPoseAnalyzer.calculate_body_orientation(landmarks, mp_pose),
            'limb_lengths': EnhancedPoseAnalyzer.calculate_limb_lengths(landmarks, mp_pose),
            'raw_positions': EnhancedPoseAnalyzer.extract_raw_positions(landmarks, mp_pose)
        }
        
        return pose_data
    
    @staticmethod
    def calculate_body_center(landmarks, mp_pose):
        """Calculate the center of the body"""
        # Use midpoint between shoulders and hips
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = (l_shoulder.x + r_shoulder.x + l_hip.x + r_hip.x) / 4
        center_y = (l_shoulder.y + r_shoulder.y + l_hip.y + r_hip.y) / 4
        
        return [center_x, center_y]
    
    @staticmethod
    def calculate_body_scale(landmarks, mp_pose):
        """Calculate the scale of the body relative to frame"""
        # Use distance between shoulders and hips as scale reference
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        torso_height = abs((l_shoulder.y + r_shoulder.y)/2 - (l_hip.y + r_hip.y)/2)
        
        return [shoulder_width, torso_height]
    
    @staticmethod
    def calculate_body_orientation(landmarks, mp_pose):
        """Calculate the orientation of the body (which way it's facing)"""
        # Use vector from hips to shoulders to determine orientation
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        shoulder_center = [(l_shoulder.x + r_shoulder.x)/2, (l_shoulder.y + r_shoulder.y)/2]
        hip_center = [(l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2]
        
        # Calculate angle of vector from hips to shoulders
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        orientation = np.arctan2(dy, dx)
        
        return orientation
    
    @staticmethod
    def calculate_limb_lengths(landmarks, mp_pose):
        """Calculate relative lengths of limbs"""
        # Calculate distances between joints for each limb
        limbs = {
            'left_arm': [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST
            ],
            'right_arm': [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ],
            'left_leg': [
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE
            ],
            'right_leg': [
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
        }
        
        limb_lengths = {}
        for limb_name, joints in limbs.items():
            total_length = 0
            for i in range(len(joints)-1):
                j1 = landmarks[joints[i].value]
                j2 = landmarks[joints[i+1].value]
                segment_length = np.sqrt((j2.x - j1.x)**2 + (j2.y - j1.y)**2)
                total_length += segment_length
            limb_lengths[limb_name] = total_length
        
        return limb_lengths
    
    @staticmethod
    def extract_raw_positions(landmarks, mp_pose):
        """Extract raw positions of key landmarks"""
        key_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        positions = {}
        for landmark in key_landmarks:
            lm = landmarks[landmark.value]
            positions[landmark.name] = [lm.x, lm.y, lm.z] if hasattr(lm, 'z') else [lm.x, lm.y]
        
        return positions