import cv2


class DanceFeedbackSystem:
    def __init__(self, pose_detector, evaluator):
        self.pose_detector = pose_detector
        self.evaluator = evaluator
        self.feedback_history = []
    
    def visualize_feedback(self, frame, evaluation):
        # Draw landmarks (if available)
        # (This would be called with pose landmarks from detection)
        
        # Display scores
        cv2.putText(frame, f"Score: {evaluation['performance_score']:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Similarity: {evaluation['reference_similarity']:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display beat information if available
        if evaluation['current_beat'] is not None:
            cv2.putText(frame, f"Beat: {evaluation['current_beat']}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display on-beat status
        if evaluation['on_beat'] is not None:
            beat_status = "On Beat!" if evaluation['on_beat'] else "Off Beat"
            color = (0, 255, 0) if evaluation['on_beat'] else (0, 0, 255)
            cv2.putText(frame, beat_status, 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display top feedback
        if evaluation['feedback']:
            for i, feedback in enumerate(evaluation['feedback'][:3]):  # Show max 3 feedback items
                cv2.putText(frame, feedback, 
                            (10, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame