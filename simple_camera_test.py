# =============================================================================
# SIMPLE CAMERA TEST - No MediaPipe (just OpenCV)
# =============================================================================

import cv2
import numpy as np
from collections import deque

# Store frame differences
frame_buffer = deque(maxlen=15)

def compute_motion_score(frames):
    """Compute motion from frames"""
    
    if len(frames) < 2:
        return 0.5
    
    total_motion = 0
    
    for i in range(len(frames) - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(frame1, frame2)
        motion = np.sum(diff) / (frame1.shape[0] * frame1.shape[1])
        total_motion += motion
    
    avg_motion = total_motion / (len(frames) - 1)
    
    # Score based on motion
    if avg_motion < 1:
        score = 0.2  # Too static
    elif avg_motion > 50:
        score = 0.3  # Too much
    else:
        score = 0.7  # Good
    
    return score, avg_motion

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("\n" + "="*70)
    print("SIMPLE MOTION DETECTION TEST")
    print("="*70)
    print("\n1. Move your head naturally")
    print("2. Wait for 15 frames")
    print("3. Result will show\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_buffer.append(frame.copy())
        
        # Display
        h, w = frame.shape[:2]
        progress = int((len(frame_buffer) / 15) * 100)
        
        cv2.putText(frame, f"Frames: {len(frame_buffer)}/15", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress bar
        cv2.rectangle(frame, (10, 80), (310, 100), (200, 200, 200), 1)
        cv2.rectangle(frame, (10, 80), (10 + int(300 * progress / 100), 100), (0, 255, 0), -1)
        cv2.putText(frame, f"{progress}%", (320, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Motion Test", frame)
        
        # When we have 15 frames
        if len(frame_buffer) == 15:
            score, motion = compute_motion_score(list(frame_buffer))
            
            print("\n" + "="*70)
            print("RESULT")
            print("="*70)
            print(f"Motion Score: {score:.2f}")
            print(f"Average Motion: {motion:.2f}")
            
            if score > 0.5:
                print("Verdict: REAL (natural movement)")
            else:
                print("Verdict: SPOOF (unnatural/static)")
            
            print("="*70 + "\n")
            
            frame_buffer.clear()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
