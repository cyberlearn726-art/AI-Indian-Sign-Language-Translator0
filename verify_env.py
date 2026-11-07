"""
Phase 1 Environment Verification Script
Universal compatibility checker for any active Python version.
Validates:
- Active Python interpreter & version
- OpenCV
- MediaPipe
- NumPy
- Scikit-Learn
- PySide6
- pyttsx3 (SAPI5 offline driver)
"""

import sys

def check_environment():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - UNIVERSAL ENVIRONMENT CHECK")
    print("=" * 65)
    
    # 1. Check Python version
    py_ver = sys.version_info
    print(f"[*] Detected Python: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    print(f"[*] Executable Path: {sys.executable}")
    
    all_ok = True

    # 2. Check OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV (cv2): {cv2.__version__}")
    except ImportError as e:
        print(f"[X] OpenCV missing: {e}")
        all_ok = False

    # 3. Check MediaPipe
    try:
        import mediapipe as mp
        print(f"[OK] MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"[X] MediaPipe missing: {e}")
        print("    Tip: If 'pip install mediapipe' fails on very new Python releases,")
        print("    ensure your pip is updated with 'python -m pip install --upgrade pip'.")
        all_ok = False

    # 4. Check NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy: {np.__version__}")
    except ImportError as e:
        print(f"[X] NumPy missing: {e}")
        all_ok = False

    # 5. Check Scikit-Learn & Joblib
    try:
        import sklearn
        import joblib
        print(f"[OK] Scikit-Learn: {sklearn.__version__}")
        print(f"[OK] Joblib: {joblib.__version__}")
    except ImportError as e:
        print(f"[X] Scikit-Learn or Joblib missing: {e}")
        all_ok = False

    # 6. Check PySide6
    try:
        import PySide6
        from PySide6.QtCore import qVersion
        print(f"[OK] PySide6: {PySide6.__version__} (Qt {qVersion()})")
    except ImportError as e:
        print(f"[X] PySide6 missing: {e}")
        all_ok = False

    # 7. Check Offline pyttsx3 (Windows SAPI5)
    try:
        import pyttsx3
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        print(f"[OK] pyttsx3 (SAPI5): Detected {len(voices)} offline voice(s)")
        for idx, voice in enumerate(voices):
            print(f"     - Voice {idx + 1}: {voice.name}")
    except Exception as e:
        print(f"[X] pyttsx3 / SAPI5 initialization error: {e}")
        all_ok = False

    print("=" * 65)
    if all_ok:
        print(">>> ALL DEPENDENCIES ARE PROPERLY INSTALLED AND READY! <<<")
    else:
        print(">>> TO INSTALL ALL MISSING PACKAGES, RUN:")
        print("    pip install -r requirements.txt")
    print("=" * 65)
    return all_ok

if __name__ == "__main__":
    check_environment()
