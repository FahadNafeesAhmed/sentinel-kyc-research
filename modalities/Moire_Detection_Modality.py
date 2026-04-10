import cv2
import numpy as np
from typing import Dict

def detect_moire_patterns(frame: np.ndarray) -> Dict:
    """Detects screen artifacts using 2D Fast Fourier Transform (FFT)"""
    if frame is None:
        return {"detected": False, "score": 0.0}

    # 1. Preprocess: Resize and grayscale for speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    
    # 2. Compute 2D FFT
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 3. Analyze High-Frequency Components
    # Screens produce periodic peaks outside the center (low-freq) area
    rows, cols = resized.shape
    crow, ccol = rows // 2, cols // 2
    
    # Mask out the center (DC component and low frequencies)
    magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10] = 0
    
    # Calculate score based on peak intensity in high-frequency regions
    max_val = np.max(magnitude_spectrum)
    mean_val = np.mean(magnitude_spectrum)
    
    # Screen patterns typically show as sharp isolated peaks
    score = (max_val / (mean_val + 1))
    
    return {
        "score": float(score),
        "detected": score > 5.0, # Threshold for moire presence
        "magnitude_map": magnitude_spectrum
    }
