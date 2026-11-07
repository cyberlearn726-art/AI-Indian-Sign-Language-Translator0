"""
Interactive ISL Landmark Dataset Collection Tool (Standard Library CSV)
Records 126-dimensional normalized MediaPipe landmark feature vectors.
Supports both single-hand and two-hand Indian Sign Language signs.
"""

import sys
import os
import csv
import json
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


def load_classes():
    """Loads class definitions from JSON configuration."""
    if not config.CLASSES_CONFIG_PATH.exists():
        return []
    with open(config.CLASSES_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("classes", [])


def save_samples_to_csv(samples_data: list, label: str):
    """
    Appends recorded feature vectors to dataset/raw_landmarks.csv.
    Each row contains 126 normalized float features + 1 label column.
    """
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = config.RAW_DATA_PATH

    feature_cols = [f"f_{i}" for i in range(config.TOTAL_FEATURES)]
    header = feature_cols + ["label"]

    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for feat in samples_data:
            writer.writerow(list(feat) + [label])

    print(f"[Dataset] Saved {len(samples_data)} samples for sign '{label}' to {csv_file.name}.")


def run_data_collector():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - DATASET COLLECTOR")
    print("=" * 65)

    classes = load_classes()
    if not classes:
        print("[!] No classes found in sign_classes.json.")
        return

    print("Available Sign Classes to Record:")
    for c in classes:
        print(f"  [{c['id']:2d}] {c['name']:10s} ({c['type']}, {c['hands']} hand(s)) - {c['description']}")
    print("=" * 65)

    try:
        selection = input("\nEnter Class ID or Sign Name to record (or 'q' to quit): ").strip()
    except EOFError:
        selection = "0"

    if selection.lower() == 'q':
        return

    target_class = None
    for c in classes:
        if str(c["id"]) == selection or c["name"].lower() == selection.lower():
            target_class = c
            break

    if target_class is None:
        sign_label = selection.upper()
        expected_hands = 1
        print(f"[*] Creating custom sign: '{sign_label}'")
    else:
        sign_label = target_class["name"]
        expected_hands = target_class["hands"]
        print(f"[*] Selected: '{sign_label}' (Requires {expected_hands} hand(s))")
        print(f"    Description: {target_class['description']}")

    target_samples = 60
    recorded_samples = []

    camera = CameraStream()
    if not camera.start():
        print("[!] Failed to open camera.")
        return

    detector = HandDetector()

    window_name = f"ISL Data Collector - Recording: {sign_label}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    STATE_IDLE = 0
    STATE_COUNTDOWN = 1
    STATE_RECORDING = 2

    state = STATE_IDLE
    countdown_start = 0.0
    countdown_seconds = 3

    print("\n[*] Controls:")
    print("    - Press 's' or SPACE: Start 3-second countdown to record")
    print("    - Press 'q' or ESC: Exit collector")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            mirrored = cv2.flip(frame, 1)
            detection_res, annotated, features = detector.process_frame(mirrored)

            h, w = annotated.shape[:2]
            num_detected = len(detection_res.hand_landmarks) if detection_res.hand_landmarks else 0

            # HUD Background Header
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 75), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)

            # Header info
            cv2.putText(annotated, f"Sign: {sign_label} | Needed Hands: {expected_hands} | Detected: {num_detected}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            if state == STATE_IDLE:
                prompt = "Press 's' or SPACE when ready to record"
                cv2.putText(annotated, prompt, (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            elif state == STATE_COUNTDOWN:
                elapsed = time.time() - countdown_start
                remaining = countdown_seconds - int(elapsed)
                if remaining > 0:
                    cd_text = f"GET READY: {remaining}..."
                    cv2.putText(annotated, cd_text, (w // 2 - 120, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3, cv2.LINE_AA)
                    cv2.putText(annotated, "Hold your hand pose firmly in frame", (w // 2 - 150, h // 2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
                else:
                    state = STATE_RECORDING
                    recorded_samples.clear()

            elif state == STATE_RECORDING:
                if num_detected >= 1:
                    recorded_samples.append(features.copy())

                progress = len(recorded_samples)
                bar_width = int((progress / target_samples) * (w - 30))

                cv2.rectangle(annotated, (15, 45), (15 + bar_width, 65), (0, 220, 0), -1)
                rec_info = f"RECORDING: {progress}/{target_samples} frames"
                cv2.putText(annotated, rec_info, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)

                if progress >= target_samples:
                    save_samples_to_csv(recorded_samples, sign_label)
                    state = STATE_IDLE
                    print(f"[OK] Completed recording {target_samples} samples for '{sign_label}'.")

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif (key == ord('s') or key == 32) and state == STATE_IDLE:
                state = STATE_COUNTDOWN
                countdown_start = time.time()

    finally:
        detector.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[*] Data collector closed.")


if __name__ == "__main__":
    run_data_collector()
