"""
Quick Calibration & Live Hand Trainer with Data Augmentation
Records live samples per sign directly from your webcam, automatically expands
them with realistic biometric variations, and trains a rock-solid, rotation-invariant model.
"""

import sys
import time
import csv
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraStream
from core.hand_detector import HandDetector
from models.train_model import train_and_evaluate
import config

TARGET_SIGNS = [
    {"name": "1", "desc": "Index finger pointing up, others closed"},
    {"name": "2", "desc": "Index and middle fingers pointing up (V shape)"},
    {"name": "3", "desc": "Thumb, index, and middle fingers up"},
    {"name": "4", "desc": "Four fingers up, thumb folded"},
    {"name": "5", "desc": "All five fingers open and spread wide"},
    {"name": "A", "desc": "TWO HANDS: Left index up, right index touches left thumb"},
    {"name": "C", "desc": "Single hand curved in C-shape"},
    {"name": "L", "desc": "Thumb and index finger forming 90-degree L-shape"},
    {"name": "GOOD", "desc": "Thumbs-up gesture held firmly"},
    {"name": "NAMASTE", "desc": "TWO HANDS: Both flat palms pressed together"}
]

SAMPLES_PER_SIGN = 45


def run_quick_train():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - QUICK CALIBRATION WIZARD")
    print("=" * 65)
    print("[*] Tip during recording: Tilt your hand slightly naturally so the AI")
    print("    learns your hand from all angles.")
    print("=" * 65)

    camera = CameraStream()
    if not camera.start():
        print("[!] Failed to open camera.")
        return

    detector = HandDetector()
    window_name = "ISL Quick Hand Calibration Wizard"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    all_recorded_rows = []
    feature_cols = [f"f_{i}" for i in range(config.TOTAL_FEATURES)]

    try:
        for idx, sign_info in enumerate(TARGET_SIGNS, 1):
            sign_name = sign_info["name"]
            desc = sign_info["desc"]

            # Phase A: 3-Second Countdown / Preparation
            countdown_start = time.time()
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue

                mirrored = cv2.flip(frame, 1)
                detection_res, annotated, feat = detector.process_frame(mirrored)
                h, w = annotated.shape[:2]

                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)

                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

                cv2.putText(annotated, f"STEP {idx}/{len(TARGET_SIGNS)}: Form Sign '{sign_name}'",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 120), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"Pose: {desc}",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

                if remaining > 0:
                    cv2.putText(annotated, f"Get Ready: {remaining}s",
                                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 180, 255), 2, cv2.LINE_AA)
                else:
                    break

                cv2.imshow(window_name, annotated)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("[*] Calibration cancelled.")
                    return

            # Phase B: Record samples while user holds pose
            samples_captured = 0
            while samples_captured < SAMPLES_PER_SIGN:
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue

                mirrored = cv2.flip(frame, 1)
                detection_res, annotated, feat = detector.process_frame(mirrored)
                num_hands = len(detection_res.hand_landmarks) if detection_res.hand_landmarks else 0

                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

                cv2.putText(annotated, f"RECORDING '{sign_name}': {samples_captured}/{SAMPLES_PER_SIGN}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, "Gently move/tilt hand slightly for best accuracy",
                            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

                if num_hands >= 1:
                    raw_feat = feat.copy()
                    all_recorded_rows.append(list(raw_feat) + [sign_name])

                    # Live Data Augmentation: create 2 realistic biometric variations per frame
                    mask = (raw_feat != 0.0).astype(np.float32)
                    noise1 = np.random.normal(0.0, 0.015, size=raw_feat.shape) * mask
                    noise2 = np.random.normal(0.0, 0.025, size=raw_feat.shape) * mask
                    all_recorded_rows.append(list(raw_feat + noise1) + [sign_name])
                    all_recorded_rows.append(list(raw_feat + noise2) + [sign_name])

                    samples_captured += 1

                bar_w = int((samples_captured / SAMPLES_PER_SIGN) * (w - 30))
                cv2.rectangle(annotated, (15, 55), (15 + bar_w, 75), (0, 220, 0), -1)

                cv2.imshow(window_name, annotated)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("[*] Calibration cancelled.")
                    return

            print(f"[OK] Calibrated '{sign_name}'.")

    finally:
        detector.close()
        camera.stop()
        cv2.destroyAllWindows()

    print("\n[*] Saving augmented dataset...")
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = feature_cols + ["label"]

    with open(config.RAW_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in all_recorded_rows:
            writer.writerow(r)

    print(f"[OK] Saved {len(all_recorded_rows)} rich biometric training samples.")
    print("[*] Retraining AI model on augmented dataset...")
    train_and_evaluate()
    print("\n>>> CALIBRATION COMPLETE! You can now run 'python app.py'! <<<")


if __name__ == "__main__":
    run_quick_train()
