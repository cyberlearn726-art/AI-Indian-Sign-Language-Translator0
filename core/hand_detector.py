"""
Hand Landmark Detection & Coordinate Normalization Engine using MediaPipe Tasks
Supports bimanual (dual-hand) tracking for Indian Sign Language (ISL).
Features:
- Canonical 2D rotation alignment (wrist to middle MCP aligned vertically)
- Scale and translation invariance
- Deterministic left-to-right spatial hand sorting
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

# Standard 21 Landmark Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),       # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky & Palm
]

COLOR_JOINTS = (0, 255, 128)
COLOR_CONNECTIONS = (255, 200, 0)
COLOR_BADGE_BG = (30, 30, 30)


def normalize_and_align_hand(coords: np.ndarray) -> np.ndarray:
    """
    Transforms 21 3D landmarks into a canonical, rotation-, translation-,
    and scale-invariant coordinate system.
    Wrist (0) -> (0, 0, 0).
    Middle MCP (9) -> strictly (0, -1, 0) along vertical axis.
    """
    wrist = coords[0].copy()
    rel = coords - wrist

    mcp9 = rel[9]
    span = np.linalg.norm(mcp9[:2])
    if span < 1e-4:
        span = np.max(np.abs(rel)) + 1e-6

    # 2D rotation to align wrist->middle MCP vector along negative Y axis (straight up)
    theta = np.arctan2(mcp9[0], -mcp9[1])
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    rot_x = rel[:, 0] * cos_t - rel[:, 1] * sin_t
    rot_y = rel[:, 0] * sin_t + rel[:, 1] * cos_t
    rot_z = rel[:, 2]

    aligned = np.stack([rot_x, rot_y, rot_z], axis=1)
    normalized = aligned / span
    return normalized.flatten()


class HandDetector:
    """
    Offline MediaPipe Hand Landmarker wrapper with canonical rotation normalization.
    """

    def __init__(self,
                 model_path: Optional[Path] = None,
                 max_num_hands: int = config.MAX_NUM_HANDS,
                 min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE):

        if model_path is None:
            model_path = config.MODELS_DIR / "hand_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model missing at: {model_path}")

        self.model_path = str(model_path)
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        Runs detection and returns raw results, annotated frame, and invariant features.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        annotated_frame = frame_bgr.copy()
        feature_vector = self.extract_features(detection_result)
        annotated_frame = self.draw_landmarks(annotated_frame, detection_result)

        return detection_result, annotated_frame, feature_vector

    def extract_features(self, detection_result) -> np.ndarray:
        """
        Extracts 126-dimensional canonical rotation-invariant feature vector.
        - Single hand: strictly placed in Slot 1 (indices 0..63).
        - Dual hands: sorted deterministically left-to-right by wrist X.
        """
        features = np.zeros(config.TOTAL_FEATURES, dtype=np.float32)

        if not detection_result.hand_landmarks:
            return features

        extracted_hands = []

        for landmarks in detection_result.hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            normalized_vec = normalize_and_align_hand(coords)
            wrist_x = coords[0, 0]  # Used for left-to-right sorting
            extracted_hands.append((wrist_x, normalized_vec))

        if len(extracted_hands) == 1:
            features[0:63] = extracted_hands[0][1]
        elif len(extracted_hands) >= 2:
            extracted_hands.sort(key=lambda item: item[0])
            features[0:63] = extracted_hands[0][1]
            features[63:126] = extracted_hands[1][1]

        return features

    def draw_landmarks(self, frame: np.ndarray, detection_result) -> np.ndarray:
        """Draws clean skeletons and glowing keypoints."""
        if not detection_result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
            pixel_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Bones
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], COLOR_CONNECTIONS, 2, cv2.LINE_AA)

            # Keypoints
            for idx, pt in enumerate(pixel_points):
                if idx in (0, 4, 8, 12, 16, 20):
                    cv2.circle(frame, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 4, COLOR_JOINTS, -1, cv2.LINE_AA)

            # Hand badge
            wrist_pt = pixel_points[0]
            label = f"Hand {hand_idx + 1}"
            if hand_idx < len(detection_result.handedness) and len(detection_result.handedness[hand_idx]) > 0:
                cat = detection_result.handedness[hand_idx][0]
                label = f"{cat.category_name} ({cat.score * 100:.0f}%)"

            bx = max(10, wrist_pt[0] - 40)
            by = min(h - 10, wrist_pt[1] + 25)
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + 110, by + 4), COLOR_BADGE_BG, -1)
            cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return frame

    def close(self):
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
