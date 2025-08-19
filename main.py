import mediapipe as mp
import cv2
import numpy as np
import json

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if (angle > 180):
        angle = 360-angle

    return angle

def normalize_landmarks(landmarks, mp_pose):
    """Normalize pose: root-center, scale by shoulder distance"""
    # Extract relevant points
    l_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    l_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    r_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    # Root-center at hips
    pelvis = (l_hip + r_hip) / 2.0

    # Scale by shoulder width
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
    if shoulder_width < 1e-5:
        shoulder_width = 1e-5  # avoid div by zero

    normalized = []
    for lm in landmarks:
        x, y = lm.x, lm.y
        normed = [(x - pelvis[0]) / shoulder_width,
                  (y - pelvis[1]) / shoulder_width]
        normalized.append(normed)
    return normalized


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Video Capture
cap = cv2.VideoCapture('test_video2.mp4')

#Webcam Capture
#cap = cv2.VideoCapture(0)

session = []

while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        #resize the frame for potrait video
        frame = cv2.resize(frame, (350, 600))

        #convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process frame for pose detection
        pose_results = pose.process(frame_rgb)
 
        #draw skeleton
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #Extract Landmarks
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            normalized_points = normalize_landmarks(landmarks, mp_pose)

            normalized_points = pose_results.pose_landmarks.landmark
            frame_data = [{"x": lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility} for lm in normalized_points]
            session.append(frame_data)

        #Locations - Put in try pass
        #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        #display the frame
        cv2.imshow('Output', frame)
    except:
        break
    
    if cv2.waitKey(1) == ord('q'):
        break

with open("test_session.json", "w") as f:
    json.dump(session, f)

cap.release()
cv2.destroyAllWindows()