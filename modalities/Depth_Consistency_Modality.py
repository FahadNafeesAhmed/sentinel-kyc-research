import numpy as np
from typing import Dict, Optional

class DepthConsistencyAnalyzer:
    def __init__(self, window_size: int = 5):
        self.prev_landmarks = None
        self.scores = []
        self.window_size = window_size

    def analyze_parallax(self, landmarks: np.ndarray) -> Dict:
        """Determines if the face is 3D by comparing Nose velocity vs Edge velocity"""
        if landmarks is None:
            return {"score": 0.0, "is_3d": False}

        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return {"score": 0.5, "is_3d": True}

        # Select points: Nose Tip (1) and Chin (152) and Right Edge (454)
        nose = landmarks[1]
        chin = landmarks[152]
        
        # Calculate motion vectors
        nose_move = np.linalg.norm(nose[:2] - self.prev_landmarks[1][:2])
        chin_move = np.linalg.norm(chin[:2] - self.prev_landmarks[152][:2])
        
        # In a 2D photo, all points move synchronously (ratio ~ 1.0)
        # In 3D, rotation causes the nose (closer) to move more than the chin relative to camera
        ratio = 1.0
        if chin_move > 0.001: # Threshold to avoid noise
            ratio = nose_move / chin_move
            
        self.prev_landmarks = landmarks
        
        # Real people usually exhibit variance between 1.2 and 2.5 during movement
        is_3d = 1.15 < ratio < 3.0
        
        return {
            "ratio": float(ratio),
            "is_3d": is_3d,
            "status": "3D DEPTH" if is_3d else "FLAT SURFACE"
        }
