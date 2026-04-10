import numpy as np
import cv2
from scipy import signal
from typing import List, Optional

FOREHEAD_INDICES = [103, 67, 109, 10, 338, 297, 332, 108, 151, 9]

class rPPGAnalyzer:
    def __init__(self, buffer_size: int = 150, fps: int = 30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.signal_buffer = [] 
        self.bpm = 0.0
        
    def extract_forehead_signal(self, frame: np.ndarray, landmarks: np.ndarray):
        """Extract average green channel intensity from forehead for pulse detection"""
        h, w, _ = frame.shape
        pts = np.array([[int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] for idx in FOREHEAD_INDICES])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        mean_intensity = cv2.mean(frame[:, :, 1], mask=mask)[0]
        
        if mean_intensity > 0:
            self.signal_buffer.append(mean_intensity)
            if len(self.signal_buffer) > self.buffer_size:
                self.signal_buffer.pop(0)
            return mean_intensity, mask
        return None, mask

    def calculate_bpm(self) -> float:
        """Estimate Heart Rate using FFT on signal buffer"""
        if len(self.signal_buffer) < self.buffer_size: return 0.0
            
        # Signal Pre-processing
        raw_signal = np.array(self.signal_buffer)
        detrended = signal.detrend(raw_signal)
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-6)
        
        # Bandpass Filter (0.7-4.0 Hz = 42-240 BPM)
        b, a = signal.butter(4, [0.7 / (self.fps / 2), 4.0 / (self.fps / 2)], btype='band')
        filtered = signal.filtfilt(b, a, normalized)
        
        # Frequency Domain Analysis
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(self.buffer_size, 1.0 / self.fps)
        
        hr_mask = (freqs > 0.7) & (freqs < 4.0)
        if not np.any(hr_mask): return 0.0
            
        peak_idx = np.argmax(fft[hr_mask])
        self.bpm = freqs[hr_mask][peak_idx] * 60.0
        return self.bpm

    def get_signal_plot_data(self) -> np.ndarray:
        return np.array(self.signal_buffer) if len(self.signal_buffer) >= 10 else np.array([])
