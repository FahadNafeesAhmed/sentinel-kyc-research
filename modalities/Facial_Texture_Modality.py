import cv2
import numpy as np
from typing import List, Tuple

FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
L_EYE_INDICES = [362, 385, 387, 263, 373, 380]
R_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]

def detect_skin_imperfections(frame: np.ndarray, landmarks: np.ndarray):
    """Detect tiny pores and acne via high-frequency skin analysis"""
    h, w, _ = frame.shape
    if landmarks is None or len(landmarks) < 468: return [], np.zeros((h,w), dtype=np.uint8)
    
    # Create Skin Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    def to_pts(indices):
        return np.array([[int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] for idx in indices])

    cv2.fillPoly(mask, [to_pts(FACE_OVAL_INDICES)], 255)
    for idxs in [L_EYE_INDICES, R_EYE_INDICES, MOUTH_INDICES]:
        cv2.fillPoly(mask, [to_pts(idxs)], 0)

    # High-Pass Filter for texture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    diff = cv2.subtract(blurred, gray)
    masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
    
    _, thresh = cv2.threshold(masked_diff, 18, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    imperfections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1 < area < 60: 
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 0.5 < radius < 10 and mask[int(y), int(x)] > 0:
                imperfections.append((int(x), int(y), int(radius)))
                
    return imperfections, masked_diff

def analyze_skin_health_score(imperfections: List[Tuple[int, int, int]]) -> float:
    """Calculates smoothness score (1.0 = smooth)"""
    return max(0.0, min(1.0, 1.0 - (len(imperfections) / 1000.0)))
