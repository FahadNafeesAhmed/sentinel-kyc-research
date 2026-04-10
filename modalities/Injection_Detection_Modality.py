import time
import numpy as np
from typing import Dict

class InjectionDetector:
    def __init__(self, window_size: int = 30):
        self.frame_times = []
        self.window_size = window_size

    def analyze_timing(self) -> Dict:
        """Analyzes jitter in frame arrival times to detect 'too perfect' virtual cameras"""
        self.frame_times.append(time.time())
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        if len(self.frame_times) < 10:
            return {"jitter": 0.0, "is_virtual": False, "status": "CALIBRATING"}

        # Calculate time difference between frames
        deltas = np.diff(self.frame_times)
        jitter = np.std(deltas)
        
        # Real hardware cameras have natural USB/sensor jitter.
        # Virtual cameras (OBS, ManyCam) often deliver frames with perfect periodicity.
        is_suspicious = jitter < 0.0008 
        
        return {
            "jitter": float(jitter),
            "is_virtual": is_suspicious,
            "status": "VIRTUAL" if is_suspicious else "PHYSICAL"
        }
