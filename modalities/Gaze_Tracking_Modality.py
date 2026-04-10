import numpy as np
import cv2
from typing import Dict, Optional

class GazeTracker:
    def __init__(self):
        # Iris indices in MediaPipe (refine_landmarks=True)
        self.L_IRIS = [474, 475, 476, 477]
        self.R_IRIS = [469, 470, 471, 472]
        self.L_EYE_OUTER = [263, 362] # Left, Right corners
        self.R_EYE_OUTER = [33, 133]

    def estimate_gaze(self, landmarks: np.ndarray) -> Dict:
        """Calculates iris position relative to eye corners to detect gaze direction"""
        if landmarks is None or len(landmarks) < 478:
            return {"direction": "CENTER", "horizontal_ratio": 0.5}

        # Calculate horizontal ratio for Left Eye
        l_iris_center = np.mean(landmarks[self.L_IRIS], axis=0)
        l_outer = landmarks[263]
        l_inner = landmarks[362]
        
        # Distance from inner corner / total eye width
        width = np.linalg.norm(l_outer - l_inner)
        dist = np.linalg.norm(l_iris_center - l_inner)
        ratio = dist / width if width > 0 else 0.5
        
        direction = "CENTER"
        if ratio < 0.35: direction = "RIGHT"
        elif ratio > 0.65: direction = "LEFT"
            
        return {
            "direction": direction,
            "ratio": float(ratio)
        }
