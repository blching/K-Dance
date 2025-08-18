import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Video Capture
cap = cv2.VideoCapture('test_video.mp4')

#Webcam Capture
#cap = cv2.VideoCapture(0)

while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        #rezie the frame for potrait videoq
        frame = cv2.resize(frame, (350, 600))

        #convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process frame for pose detection
        pose_results = pose.process(frame_rgb)
 
        #draw skeelton
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #Extract Landmarks
        landmarks = pose_results.pose_landmarks.landmark
            
        #display the frame
        cv2.imshow('Output', frame)
    except:
        break
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()