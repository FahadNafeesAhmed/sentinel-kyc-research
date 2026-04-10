import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
from modalities.Facial_Landmarks_Modality import extract_landmarks_from_frame
from modalities.Facial_Texture_Modality import detect_skin_imperfections
from modalities.Facial_rPPG_Modality import rPPGAnalyzer
from modalities.Active_Flash_Modality import ActiveFlashAnalyzer
from modalities.Gaze_Tracking_Modality import GazeTracker
from modalities.Moire_Detection_Modality import detect_moire_patterns
from modalities.Injection_Detection_Modality import InjectionDetector
from modalities.Depth_Consistency_Modality import DepthConsistencyAnalyzer
from heuristics.Liveness_Decision_Engine import compute_final_verdict

# UI Configuration
st.set_page_config(page_title="Sentinel KYC Research", layout="wide")
st.markdown("""
<style>
    .main { background: #0e1117; color: white; }
    div[data-testid="stMetricValue"] > div { color: #00F5FF; font-family: monospace; }
    .stAlert { background-color: #161b22; border: 1px solid #00F5FF; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ Analysis Controls")
cam_source = st.sidebar.text_input("Camera Source (0=webcam, or URL)", "0")
source = int(cam_source) if cam_source.isdigit() else cam_source
run_app = st.sidebar.checkbox("🚀 START CAMERA", value=False)
show_mesh = st.sidebar.checkbox("Overlay 2D face mesh", value=True)
active_flash = st.sidebar.selectbox("Flash Challenge", ["None", "RED", "BLUE"])
focus_mode = st.sidebar.selectbox("🔬 Focus Mode (Isolated)", ["Main Feed", "Texture Mask", "rPPG Region", "Moire Spectrum"])

# Layout setup
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📸 Live Research Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📈 Real-time Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        bpm_placeholder = st.empty()
        gaze_placeholder = st.empty()
    with m_col2:
        moire_placeholder = st.empty()
        spots_placeholder = st.empty()
    with m_col3:
        sensor_placeholder = st.empty()
        depth_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("⚖️ Liveness Verdict")
    verdict_placeholder = st.empty()

st.markdown("---")
st_3d_col1, st_3d_col2 = st.columns([1, 1])
with st_3d_col1:
    st.subheader("🧊 3D Mesh Reconstruction")
    mesh_3d_placeholder = st.empty()
with st_3d_col2:
    st.subheader("💓 Heart Pulse Wave")
    plot_placeholder = st.empty()

# Initialize Analyzers
rppg = rPPGAnalyzer()
flash_engine = ActiveFlashAnalyzer()
gaze_engine = GazeTracker()
injection_engine = InjectionDetector()
depth_engine = DepthConsistencyAnalyzer()

if run_app:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error(f"❌ Camera source '{source}' not found or busy.")
        run_app = False
    
    while run_app:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 1. Processing
        landmarks_data = extract_landmarks_from_frame(frame)
        moire_data = detect_moire_patterns(frame)
        injection_data = injection_engine.analyze_timing()
        
        # Isolated debug frames
        debug_frame = None
        gaze_dir, spots_count, bpm_val, is_3d = "---", 0, 0.0, False
        
        if landmarks_data:
            lms = landmarks_data["landmarks"]
            
            try:
                # A. Texture Analysis
                texture_results = detect_skin_imperfections(frame, lms)
                spots, texture_mask = texture_results if len(texture_results) == 2 else ([], np.zeros((h,w), dtype=np.uint8))
                spots_count = len(spots)
                if focus_mode == "Texture Mask": debug_frame = texture_mask
                
                # B. rPPG Analysis
                rppg_results = rppg.extract_forehead_signal(frame, lms)
                r_val, r_mask = rppg_results if len(rppg_results) == 2 else (0.0, np.zeros((h,w), dtype=np.uint8))
                if focus_mode == "rPPG Region": debug_frame = r_mask
                bpm_val = rppg.calculate_bpm()
                
                # C. Gaze & Depth
                gaze_data = gaze_engine.estimate_gaze(lms)
                gaze_dir = gaze_data.get("direction", "---")
                depth_data = depth_engine.analyze_parallax(lms)
                is_3d = depth_data.get("is_3d", False)
                
                if focus_mode == "Main Feed" and show_mesh:
                    for i, lm in enumerate(lms):
                        if i % 30 == 0: cv2.circle(frame, (int(lm[0]*w), int(lm[1]*h)), 1, (0, 255, 0), -1)
            except Exception as e:
                st.error(f"Processing error: {e}")

            # 3D Plotly Visualization
            plot_lms = lms[::5]
            fig_3d = go.Figure(data=[go.Scatter3d(x=plot_lms[:,0], y=plot_lms[:,1], z=plot_lms[:,2], mode='markers', marker=dict(size=2, color='#00F5FF'))])
            fig_3d.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            mesh_3d_placeholder.plotly_chart(fig_3d, use_container_width=True)

        # Moire Debug
        if focus_mode == "Moire Spectrum":
            debug_frame = moire_data["magnitude_map"].astype(np.uint8)

        # 2. Final Verdict Logic
        dec_results = {"bpm": bpm_val, "moire": moire_data["detected"], "sensor": injection_data["status"], "is_3d": is_3d, "spots": spots_count}
        verdict = compute_final_verdict(dec_results)
        msg = f"**{verdict['verdict']}** (Confidence: {verdict['confidence']*100:.0f}%)"
        if verdict["verdict"] == "PASSED": verdict_placeholder.success(msg, icon="🛡️")
        elif verdict["verdict"] == "SUSPICIOUS": verdict_placeholder.warning(msg, icon="⚠️")
        else: verdict_placeholder.error(msg, icon="🚫")

        # UI Update
        bpm_placeholder.metric("HEART RATE", f"{bpm_val:.1f} BPM" if bpm_val > 0 else "---")
        gaze_placeholder.metric("GAZE", gaze_dir)
        moire_placeholder.metric("MOIRE", "DETECTED" if moire_data["detected"] else "CLEAN")
        spots_placeholder.metric("TEXTURE", f"{spots_count} pts")
        sensor_placeholder.metric("SENSOR", injection_data["status"], delta=f"{injection_data['jitter']*1000:.2f}ms")
        depth_placeholder.metric("3D DEPTH", "OK" if is_3d else "FLAT")

        # Pulse Wave Plot
        if len(rppg.signal_buffer) > 1:
            sig = np.array(rppg.signal_buffer[-50:])
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
            fig = go.Figure(data=go.Scatter(y=sig, mode='lines', line=dict(color='#00F5FF', width=2)))
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False)
            plot_placeholder.plotly_chart(fig, use_container_width=True)

        # Output to Streamlit (Swap feed if debug mode is active)
        final_video = debug_frame if debug_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(final_video)
        if not run_app: break
    cap.release()
else:
    st.info("Enable camera in sidebar to start analysis.")
