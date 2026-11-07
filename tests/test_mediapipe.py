"""
Phase 3 Verification Test: Hand Detection & Invariant Landmark Extraction
Connects CameraStream with MediaPipe HandDetector.
Supports real-time bimanual (dual-hand) tracking for Indian Sign Language.
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
import config


def run_mediapipe_test():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PHASE 3 MEDIAPIPE TEST")
    print("=" * 65)
    print("[*] Initializing CameraStream...")
    camera = CameraStream(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        target_fps=config.TARGET_FPS
    )

    if not camera.start():
        print("[!] ERROR: Could not start camera stream.")
        return False

    print("[*] Initializing MediaPipe Hand Landmarker (Dual-Hand Mode)...")
    try:
        detector = HandDetector()
        print("[OK] MediaPipe detector successfully loaded offline model.")
    except Exception as e:
        print(f"[!] ERROR initializing MediaPipe HandDetector: {e}")
        camera.stop()
        return False

    window_name = "ISL Translator - Phase 3 Hand Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps_display = 0.0

    print("[*] Hand tracker active. Show 1 or 2 hands to your camera.")
    print("[*] Press 'q' or 'ESC' to exit.")
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

            # Mirror frame horizontally for natural user interaction
            mirrored_frame = cv2.flip(frame, 1)

            # Process frame with MediaPipe
            detection_res, annotated_frame, features = detector.process_frame(mirrored_frame)

            num_detected = len(detection_res.hand_landmarks) if detection_res.hand_landmarks else 0

            # UI HUD Overlay
            h, w = annotated_frame.shape[:2]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

            # Status title
            cv2.putText(annotated_frame, "AI ISL Translator - Phase 3: Hand Tracking",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

            # Hand count color coding: Green for 2 hands (ISL ready), Cyan for 1 hand, Orange for 0
            count_color = (0, 255, 0) if num_detected == 2 else ((255, 200, 0) if num_detected == 1 else (0, 140, 255))
            hand_status = f"Hands: {num_detected} / 2"

            # Telemetry text
            telemetry = f"FPS: {fps_display:.1f} | {hand_status} | Feature Vector: {len(features)} dims | 'q' to Quit"
            cv2.putText(annotated_frame, telemetry,
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, count_color, 1, cv2.LINE_AA)

            # Show visual instruction when no hands detected
            if num_detected == 0:
                guide_text = "Place one or both hands in frame"
                cv2.putText(annotated_frame, guide_text,
                            (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Exit key pressed by user.")
                break

    finally:
        print("[*] Cleaning up detector and camera...")
        detector.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Phase 3 Hand Detection test completed successfully.")

    return True


if __name__ == "__main__":
    run_mediapipe_test()
