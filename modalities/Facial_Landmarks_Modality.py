import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from scipy.spatial import distance as dist

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ROI indices for EAR/MAR
L_EYE_INDICES = [362, 385, 387, 263, 373, 380]
R_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    logger.error(f"MediaPipe failed: {e}")

def calculate_ear(landmarks: np.ndarray, eye_indices: List[int]) -> float:
    """Computes Eye Aspect Ratio"""
    v1 = dist.euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = dist.euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h = dist.euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

def calculate_mar(landmarks: np.ndarray) -> float:
    """Computes Mouth Aspect Ratio"""
    v = dist.euclidean(landmarks[13], landmarks[14])
    h = dist.euclidean(landmarks[78], landmarks[308])
    return v / h if h != 0 else 0.0

def extract_landmarks_from_frame(frame: np.ndarray) -> Optional[Dict]:
    """Extraction of 478 points and basic ratios"""
    if frame is None or not MEDIAPIPE_AVAILABLE: return None
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks: return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)
        
        l_ear = calculate_ear(landmarks, L_EYE_INDICES)
        r_ear = calculate_ear(landmarks, R_EYE_INDICES)
        return {
            "landmarks": landmarks,
            "ear": (l_ear + r_ear) / 2.0,
            "mar": calculate_mar(landmarks),
            "l_ear": l_ear,
            "r_ear": r_ear
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def analyze_liveness_indicators(sequence_data: List[Dict]) -> Dict:
    """Analyze blink and mouth trends"""
    if not sequence_data: return {"score": 0.0, "events": []}
    
    ears = [d["ear"] for d in sequence_data]
    mars = [d["mar"] for d in sequence_data]
    
    blink_count = 0
    blink_threshold = 0.22
    for i in range(1, len(ears) - 1):
        if ears[i] < blink_threshold and ears[i-1] >= blink_threshold:
            blink_count += 1
            
    max_mar = np.max(mars) if mars else 0
    mouth_activity = np.std(mars) if mars else 0
    
    score = 0.4
    if blink_count > 0: score += 0.3
    if mouth_activity > 0.05: score += 0.2
    if max_mar > 0.4: score += 0.1
    
    return {
        "score": min(1.0, score),
        "blink_count": blink_count,
        "max_mar": float(max_mar)
    }

def get_modality_status() -> bool:
    return MEDIAPIPE_AVAILABLE
