import cv2
import numpy as np
import mediapipe as mp

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