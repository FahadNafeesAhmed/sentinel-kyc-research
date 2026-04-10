import cv2
import numpy as np
from typing import Optional, Dict

class ActiveFlashAnalyzer:
    def __init__(self):
        self.reference_color = None
        self.flash_active = False
        self.expected_color = None # "RED", "BLUE", etc.

    def analyze_reflection(self, frame: np.ndarray, landmarks: np.ndarray, current_flash: Optional[str]) -> Dict:
        """Checks if skin color shifts towards the UI flash color"""
        if landmarks is None or current_flash is None:
            return {"score": 0.0, "match": False}

        h, w, _ = frame.shape
        # ROI: Forehead and Cheeks are best for reflection
        roi_indices = [103, 67, 10, 338, 297, 117, 346] 
        pts = np.array([[int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] for idx in roi_indices])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        
        # Logic: If flash is RED, R channel should increase relative to others
        # This is a simplified research heuristic
        score = 0.0
        if current_flash == "RED":
            # Check if Red is dominant or increased
            score = mean_bgr[2] / (mean_bgr[0] + mean_bgr[1] + 1)
        elif current_flash == "BLUE":
            score = mean_bgr[0] / (mean_bgr[1] + mean_bgr[2] + 1)
            
        return {
            "score": float(score),
            "match": score > 0.5, # Simple threshold for research
            "mean_bgr": mean_bgr
        }
