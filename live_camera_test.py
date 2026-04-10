import cv2
import numpy as np
import time
from modalities.Facial_Landmarks_Modality import extract_landmarks_from_frame, L_EYE_INDICES, R_EYE_INDICES
from modalities.Facial_Texture_Modality import detect_skin_imperfections
from modalities.Facial_rPPG_Modality import rPPGAnalyzer

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 1. Initialize rPPG Modality
    rppg = rPPGAnalyzer(buffer_size=150, fps=30)
    
    print("Camera Test Started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 1. Extract Landmarks
        data = extract_landmarks_from_frame(frame)
        
        if data:
            landmarks = data["landmarks"]
            
            # 2. Detect skin features (Spores)
            imperfections = detect_skin_imperfections(frame, landmarks)
            
            # 3. Detect rPPG (Heart Rate) - Forehead Region
            rppg.extract_forehead_signal(frame, landmarks)
            bpm = rppg.calculate_bpm()
            
            # --- DRAWING ---
            
            # Draw Skin imperfections
            for (x, y, r) in imperfections:
                cv2.circle(frame, (x, y), 1, (255, 255, 0), 1)
            
            # Draw Landmarks
            for i, lm in enumerate(landmarks):
                # Sample 468 points for performance
                if i % 30 == 0:
                    px, py = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
            
            # Display Features
            cv2.putText(frame, f"Spots: {len(imperfections)}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # BPM Display (Needs enough data in buffer)
            if bpm > 0:
                cv2.putText(frame, f"HEART RATE: {bpm:.1f} BPM", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                progress = int(len(rppg.signal_buffer) / 150 * 100)
                cv2.putText(frame, f"rPPG Loading: {progress}%", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(frame, f"EAR: {data['ear']:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        else:
            cv2.putText(frame, "NO FACE DETECTED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Sentinel KYC - Multi-Modality Research", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
