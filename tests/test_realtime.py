"""
Phase 6 Verification Test: Real-Time Indian Sign Language Recognition
Fuses CameraStream, MediaPipe HandDetector, ISLClassifier, and GestureRecognizer.
Displays live video with skeleton, confidence meters, stability gauges, and clean exhibition HUD.
Press 'q' or 'ESC' to exit.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraStream
from core.hand_detector import HandDetector
from core.classifier import ISLClassifier
from core.recognizer import GestureRecognizer
import config


def run_realtime_recognition():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PHASE 6 REAL-TIME RECOGNITION")
    print("=" * 65)

    # 1. Initialize Camera
    print("[*] Initializing CameraStream...")
    camera = CameraStream()
    if not camera.start():
        print("[!] ERROR: Could not start webcam.")
        return False

    # 2. Initialize MediaPipe Detector
    print("[*] Initializing MediaPipe Hand Detector (Dual-Hand Mode)...")
    try:
        detector = HandDetector()
    except Exception as e:
        print(f"[!] Error loading MediaPipe detector: {e}")
        camera.stop()
        return False

    # 3. Initialize Trained Classifier
    print("[*] Loading trained ISLClassifier...")
    try:
        classifier = ISLClassifier()
        print(f"[OK] Classifier loaded with {len(classifier.classes)} classes.")
    except Exception as e:
        print(f"[!] Error loading classifier: {e}")
        detector.close()
        camera.stop()
        return False

    # 4. Initialize Temporal Gesture Recognizer
    recognizer = GestureRecognizer()

    window_name = "AI Indian Sign Language Translator - Live Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps_display = 0.0
    confirmed_history = []

    print("[*] Starting live recognition loop...")
    print("[*] Show your hand signs in front of the camera!")
    print("[*] Controls: Press 'q' or 'ESC' to exit, 'c' to clear history.")
    print("=" * 65)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # Mirror frame horizontally for intuitive interaction
            mirrored = cv2.flip(frame, 1)
            h, w = mirrored.shape[:2]

            # 1. Check lighting
            lighting_msg, is_light_ok = recognizer.evaluate_lighting(mirrored)

            # 2. Extract hands & landmarks
            detection_res, annotated, features = detector.process_frame(mirrored)
            num_hands = len(detection_res.hand_landmarks) if detection_res.hand_landmarks else 0

            # 3. Classify gestures
            raw_sign, raw_conf, candidates = classifier.predict(features)

            # 4. Temporal consensus & debounce filtering
            state = recognizer.process(raw_sign, raw_conf)

            # Check for new confirmed sign emission
            if state["new_emission"]:
                confirmed_history.append(state["new_emission"])
                if len(confirmed_history) > 6:
                    confirmed_history.pop(0)
                print(f"[CONFIRMED SIGN] >>> {state['new_emission']} (Confidence: {state['confidence'] * 100:.1f}%)")

            # 5. Render Clean Exhibition HUD Overlay
            # Top header bar (semi-transparent dark)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 20), -1)
            # Bottom footer bar
            cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

            # Row 1: Title (Left) and Telemetry (Right) - zero overlap
            cv2.putText(annotated, "AI ISL TRANSLATOR",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

            hand_color = (0, 255, 0) if num_hands == 2 else ((0, 220, 255) if num_hands == 1 else (160, 160, 160))
            cv2.putText(annotated, f"FPS: {fps_display:.1f} | Hands: {num_hands}/2",
                        (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1, cv2.LINE_AA)

            # Row 2: Recognized Sign Display (Large clear text)
            display_sign = state["display_sign"]
            is_stable = state["is_stable"]
            sign_color = (0, 255, 0) if is_stable else ((0, 220, 255) if num_hands > 0 else (170, 170, 170))

            cv2.putText(annotated, f"Sign: {display_sign}",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.92, sign_color, 2, cv2.LINE_AA)

            # Row 3: Confidence, Stability Bar, Lighting Status
            conf_pct = state["confidence"] * 100
            conf_str = f"Confidence: {conf_pct:.1f}%" if num_hands > 0 else "Confidence: --"
            cv2.putText(annotated, conf_str,
                        (15, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

            # Stability Progress Meter Bar
            bar_x = 195
            bar_y = 90
            bar_w = 140
            bar_h = 14
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
            fill_w = int(bar_w * state["stability_ratio"])
            fill_color = (0, 255, 0) if is_stable else (0, 180, 255)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
            cv2.putText(annotated, "Stability", (bar_x + bar_w + 8, bar_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            # Lighting indicator on far right of Row 3
            light_color = (0, 255, 120) if is_light_ok else (0, 140, 255)
            cv2.putText(annotated, lighting_msg,
                        (w - 180, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, light_color, 1, cv2.LINE_AA)

            # Footer: Confirmed Signs Stream
            history_str = "Confirmed: " + (" ".join(confirmed_history) if confirmed_history else "None yet")
            cv2.putText(annotated, history_str,
                        (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, "Press 'c' to clear | 'q' to exit",
                        (w - 230, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                confirmed_history.clear()
                recognizer.reset()
                print("[*] Cleared confirmed signs history.")

    finally:
        print("[*] Shutting down real-time recognition...")
        detector.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Phase 6 Real-Time Recognition completed.")

    return True


if __name__ == "__main__":
    run_realtime_recognition()
