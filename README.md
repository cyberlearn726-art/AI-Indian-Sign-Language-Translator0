# 🤟 AI-Powered Indian Sign Language (ISL) Translator

An exhibition-ready, real-time desktop application that translates **Indian Sign Language (ISL)** gestures into readable English text and spoken audio. 

Built with **Python**, **OpenCV**, **MediaPipe**, **Scikit-Learn**, and **PySide6 (Qt)**. Operates **100% offline** without any internet connection, cloud APIs, or external server dependencies.

---

## 🌟 Key Features

- **Real-Time Bimanual Tracking**: Tracks up to 2 hands simultaneously using MediaPipe Hands (42 3D keypoints total) to support authentic two-handed Indian Sign Language signs.
- **Canonical Rotation Normalization**: Automatically calculates hand orientation and normalizes coordinates along the vertical axis. Gestures remain recognized even if your wrist or hand is tilted.
- **Scale & Translation Invariant**: Normalizes landmark positions by palm span and wrist origin, functioning reliably across different distances from the camera.
- **Temporal Consensus & Smoothing**: Uses a rolling-window consensus filter to eliminate frame-by-frame flicker and accidental hand-transition triggers.
- **100% Offline Speech Synthesis**: Uses native Microsoft Windows SAPI5 (`David` / `Zira` voices) on a dedicated non-blocking thread so the camera feed and GUI never drop frames.
- **Modern Dark Desktop GUI**: High-contrast, exhibition-grade PySide6 interface with live skeleton visualizer, confidence meter, hold stability gauge, constructed sentence builder, and translation history log.
- **Portable USB Deployment**: Includes an automated packaging script to build a standalone Windows `.exe` that runs from a pendrive without needing Python or VS Code.

---

## 🖐️ Supported ISL Signs

| Category | Signs | Hand Mode | Description |
| :--- | :--- | :--- | :--- |
| **Numbers** | `1`, `2`, `3`, `4`, `5` | Single Hand | Canonical ISL number counting gestures |
| **Alphabets** | `A` | **Both Hands** | Authentic ISL: Left index upright, Right index touches Left thumb |
| **Alphabets** | `C`, `L` | Single Hand | `C` (Curved arc), `L` (90° thumb and index angle) |
| **Vocabulary** | `GOOD` | Single Hand | Thumbs-up gesture held firmly |
| **Vocabulary** | `NAMASTE` | **Both Hands** | Both flat open palms pressed together vertically |

*(Additional signs can be added in 1 minute using the built-in Quick Calibration Wizard).*

---

## 📋 System Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Python**: Python 3.10, 3.11, or 3.12 (64-bit).
- **Webcam**: Standard built-in laptop camera or external USB webcam.
- **Internet**: **None required** during operation.

---

## 🚀 Installation & Setup Guide (For New Users)

Follow these steps to set up and run the project on any Windows PC:

### Step 1: Clone or Download the Repository

```powershell
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
```

*(Or click **Code -> Download ZIP** on GitHub and extract the folder).*

---

### Step 2: Create a Virtual Environment (Recommended)

Open **PowerShell** or **Command Prompt** inside the project folder:

```powershell
# Create an isolated virtual environment
python -m venv venv

# Activate it:
# PowerShell:
.\venv\Scripts\Activate.ps1
# (If execution policy blocks script, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process, then re-run)

# Command Prompt (cmd):
# .\venv\Scripts\activate.bat
```

---

### Step 3: Install Required Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Verify the Environment

Run the diagnostic script to ensure all computer vision and speech components are ready:

```powershell
python verify_env.py
```

---

## 🖥️ How to Use the Application

### 1. Launch the Desktop App
```powershell
python app.py
```

### 2. Main Controls
- Click **▶ Start Camera** to open your webcam feed.
- Perform any supported sign in front of the camera:
  - **`1`**, **`2`**, **`3`**, **`4`**, **`5`**: Hold up counting fingers.
  - **`GOOD`**: Hold a thumbs-up.
  - **`NAMASTE`**: Bring both flat palms together in front of your chest.
  - **`A`**: Point left index up, touch left thumb with right index finger.
- Hold the gesture steady for $\approx 0.2$ seconds. The **Hold Stability** bar fills up and locks in the sign.
- The confirmed sign is automatically added to the **Current Translation** box.
- Click **🔊 Speak Translation** to hear the offline voice speak your sentence!
- Use **␣ Space**, **⌫ Backspace**, or **🗑 Clear** to edit text.
- Click **🔊 Speak Selected** in the **Translation History** panel to re-speak any past translation.

---

## ⚡ 1-Minute Custom Hand Calibration (Optional)

If you have unique hand proportions or different lighting and want **99%+ accuracy** calibrated to your exact hands:

```powershell
python quick_train.py
```

1. The wizard steps through each sign (`1`, `2`, `3`, `4`, `5`, `A`, `C`, `L`, `GOOD`, `NAMASTE`).
2. Shows a **3-second countdown** for each sign.
3. Automatically captures your hand from your webcam, applies biometric data augmentation, and **retrains the model in under 1 second**.
4. Relaunch `python app.py` for customized accuracy!

---

## 📦 How to Package as a Portable Pendrive EXE (Zero-Python Mode)

To deploy the app to a USB pendrive for a science exhibition or demo on laptops without Python installed:

```powershell
python build_exe.py
```

This creates a standalone distribution folder:
```text
dist/Sign_Language_Translator/
    ├── SignLanguageTranslator.exe
    ├── models/
    │   ├── hand_landmarker.task
    │   ├── isl_classifier.joblib
    │   └── label_encoder.joblib
    ├── dataset/
    │   └── sign_classes.json
    └── README.txt
```

**To use on another computer:**
1. Copy the `Sign_Language_Translator` folder to a USB pendrive.
2. Plug the USB into any Windows 10/11 laptop.
3. Double-click `SignLanguageTranslator.exe` to run immediately!

---

## 📂 Project Architecture

```text
ISL_Translator/
│
├── app.py                    # Main PySide6 desktop application launcher
├── config.py                 # Central configuration (thresholds, resolutions, timings)
├── quick_train.py            # 1-minute live webcam calibration wizard
├── build_exe.py              # PyInstaller portable packaging builder
├── verify_env.py             # System diagnostic check
├── requirements.txt          # Frozen, verified dependencies
│
├── core/
│   ├── camera.py             # Low-latency camera capture manager
│   ├── hand_detector.py      # MediaPipe bimanual tracker & rotation normalizer
│   ├── classifier.py         # Multi-class inference engine with confidence scoring
│   ├── recognizer.py         # Temporal consensus smoothing & debounce state machine
│   ├── translator.py         # Sentence accumulator, punctuation & editing engine
│   └── speech_engine.py      # Non-blocking native Windows SAPI5 voice synthesizer
│
├── gui/
│   ├── main_window.py        # PySide6 modern UI window with History log
│   └── styles.py             # Exhibition dark theme QSS stylesheet
│
├── dataset/
│   ├── raw_landmarks.csv     # Recorded/augmented landmark training dataset
│   ├── sign_classes.json     # Metadata and descriptions for supported signs
│   ├── collect_data.py       # Interactive dataset recording utility
│   ├── generate_baseline_data.py # Canonical dataset generator
│   └── verify_dataset.py     # Dataset integrity & balance checker
│
├── models/
│   ├── hand_landmarker.task  # Offline MediaPipe hand landmark model
│   ├── isl_classifier.joblib # Production trained Random Forest model
│   ├── label_encoder.joblib  # Class label encoder
│   └── train_model.py        # Model training and validation script
│
└── tests/
    ├── test_camera.py        # Camera integration test
    ├── test_mediapipe.py     # Hand tracking test
    ├── test_classifier.py    # Offline model benchmark test
    ├── test_realtime.py      # HUD recognition benchmark test
    └── test_translation_speech.py # Speech & translation test
```

---

## 📜 Ethical Note & Linguistic Transparency

Recognizing static signs and fingerspelling is an important foundation for assistive computer vision, but it is not an exhaustive replacement for natural Indian Sign Language grammar, spatial syntax, and facial expressions. This project transparently recognizes supported vocabulary and fingerspelling subsets for educational and exhibition demonstrations.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
