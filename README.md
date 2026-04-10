# Sentinel KYC: Multi-Modal Liveness Research

![Sentinel KYC Architecture](docs/sentinel_kyc_architecture.png)

Sentinel KYC is a robust, research-grade liveness detection and anti-spoofing framework designed to distinguish between real human subjects and various presentation attacks (spoofing). It utilizes a multi-modal approach combining geometric, physiological, and frequency-domain analysis.

## 🚀 Key Features
- **rPPG Physiological Extraction**: Detects heart rate (BPM) from facial skin variance using FFT.
- **Moiré Pattern Detection**: Identifies "screen-on-screen" re-filming attacks via 2D Fast Fourier Transform.
- **3D Depth Consistency**: Analyzes motion parallax to distinguish between 2D photos and 3D faces.
- **Injection Detection**: Detects virtual cameras (OBS, ManyCam) via frame arrival jitter analysis.
- **Active Challenge-Response**: Integrated gaze tracking and active color flash reflection testing.
- **Real-time 3D Mesh**: Live visualization of 478 MediaPipe landmarks in 3D space.

## 🛠️ Tech Stack
- **Core**: Python 3.10+, OpenCV, MediaPipe
- **Mathematics**: NumPy, SciPy (Butterworth Filter, FFT)
- **UI & Viz**: Streamlit, Plotly
- **Documentation**: LaTeX (TikZ)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sentinel-kyc-research.git
   cd sentinel-kyc-research
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

1. **Start the Research Dashboard**:
   ```bash
   streamlit run research_app.py
   ```
2. **Select Camera Source**: 
   - Enter `0` for default webcam.
   - Enter your DroidCam / IP Camera URL (e.g., `http://192.168.1.x:4747/video`).
3. **Analyze**: Use the **"Focus Mode"** sidebar to inspect isolated signals (FFT Spectrum, Texture Mask, etc.).

## 📖 Documentation
Complete technical documentation including algorithmic derivations and threat models is available in the `docs/` folder in LaTeX format.

---
**Author**: Fahad Nafees  
**Project**: Independent Research on Liveness and Anti-Spoofing.
