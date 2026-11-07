"""
Automated PyInstaller Packaging Script for Portable USB Pendrive Deployment
Builds a standalone, zero-installation portable Windows folder:
dist/Sign_Language_Translator/
    ├── SignLanguageTranslator.exe
    ├── models/
    │   ├── hand_landmarker.task
    │   ├── isl_classifier.joblib
    │   └── label_encoder.joblib
    ├── dataset/
    │   └── sign_classes.json
    └── README.txt
"""

import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_FOLDER = DIST_DIR / "Sign_Language_Translator"


def build_portable_package():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PORTABLE USB EXE BUILDER")
    print("=" * 65)

    # 1. Clean previous builds
    if DIST_DIR.exists():
        print("[*] Cleaning previous dist/ directory...")
        shutil.rmtree(DIST_DIR, ignore_errors=True)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    # 2. Formulate PyInstaller command
    entry_script = PROJECT_ROOT / "app.py"
    exe_name = "SignLanguageTranslator"

    # Add required data folders
    models_src = PROJECT_ROOT / "models"
    dataset_src = PROJECT_ROOT / "dataset"

    pyinstaller_args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",                       # Folder distribution (fast startup, USB friendly)
        "--windowed",                     # Hide black console window
        "--name", exe_name,
        "--add-data", f"{models_src};models",
        "--add-data", f"{dataset_src / 'sign_classes.json'};dataset",
        # Hidden imports to ensure runtime completeness
        "--hidden-import", "sklearn",
        "--hidden-import", "sklearn.ensemble._forest",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.special",
        "--hidden-import", "win32com",
        "--hidden-import", "win32com.client",
        "--hidden-import", "pythoncom",
        "--hidden-import", "pyttsx3",
        "--hidden-import", "pyttsx3.drivers",
        "--hidden-import", "pyttsx3.drivers.sapi5",
        "--hidden-import", "mediapipe",
        "--hidden-import", "mediapipe.tasks",
        "--hidden-import", "mediapipe.tasks.python",
        "--hidden-import", "mediapipe.tasks.python.vision",
        "--hidden-import", "PySide6",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtWidgets",
        str(entry_script)
    ]

    print("[*] Launching PyInstaller packaging engine...")
    print(f"[*] Command: {' '.join(pyinstaller_args[:6])} ...")

    result = subprocess.run(pyinstaller_args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("[!] PyInstaller packaging encountered an error.")
        return False

    # 3. Create clean final structure: dist/Sign_Language_Translator
    raw_dist = DIST_DIR / exe_name
    if raw_dist.exists() and raw_dist != OUTPUT_FOLDER:
        if OUTPUT_FOLDER.exists():
            shutil.rmtree(OUTPUT_FOLDER)
        raw_dist.rename(OUTPUT_FOLDER)

    # 4. Copy models and configs directly into portable folder if missing
    portable_models = OUTPUT_FOLDER / "models"
    portable_models.mkdir(parents=True, exist_ok=True)
    for model_file in ["hand_landmarker.task", "isl_classifier.joblib", "label_encoder.joblib"]:
        src_file = models_src / model_file
        if src_file.exists():
            shutil.copy2(src_file, portable_models / model_file)

    portable_dataset = OUTPUT_FOLDER / "dataset"
    portable_dataset.mkdir(parents=True, exist_ok=True)
    if (dataset_src / "sign_classes.json").exists():
        shutil.copy2(dataset_src / "sign_classes.json", portable_dataset / "sign_classes.json")

    # 5. Write Exhibition README.txt for USB users
    readme_content = """================================================================================
AI-POWERED INDIAN SIGN LANGUAGE (ISL) TRANSLATOR
Exhibition-Ready Portable Offline Edition
================================================================================

Target OS: Windows 10 / Windows 11 (64-bit)
Requirements: Built-in or external USB Webcam
Internet: ZERO INTERNET NEEDED. Completely offline.
Python / VS Code: NOT REQUIRED on the host computer.

--------------------------------------------------------------------------------
HOW TO RUN FROM A USB PENDRIVE:
--------------------------------------------------------------------------------
1. Copy the entire 'Sign_Language_Translator' folder to your USB pendrive.
2. Plug the USB pendrive into any Windows 10 or Windows 11 computer.
3. Double-click 'SignLanguageTranslator.exe' inside this folder.
4. Click '▶ Start Camera' on the desktop interface.
5. Perform supported Indian Sign Language signs in front of the camera:
   - '1', '2', '3', '4', '5' (Numbers)
   - 'A', 'C', 'L' (Alphabets)
   - 'GOOD' (Thumbs up)
   - 'NAMASTE' (Both flat palms pressed together)
6. Click '🔊 Speak Translation' to hear the offline computer voice!

--------------------------------------------------------------------------------
TROUBLESHOOTING:
--------------------------------------------------------------------------------
- Camera not opening: Ensure Windows Settings -> Privacy & Security -> Camera
  has 'Allow desktop apps to access your camera' toggled ON.
- Audio / Speech: Ensure laptop speakers or headphones are turned on.
================================================================================
"""
    with open(OUTPUT_FOLDER / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("=" * 65)
    print(f"[OK] PORTABLE APPLICATION PACKAGE CREATED AT:")
    print(f"     {OUTPUT_FOLDER.resolve()}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    build_portable_package()
