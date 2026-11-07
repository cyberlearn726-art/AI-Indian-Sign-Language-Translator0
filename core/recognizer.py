"""
Temporal Gesture Recognition & Stabilization Engine
Implements:
- Lighting & exposure quality analysis
- Rolling window consensus smoothing (eliminates frame jitter)
- Stability hold counter (ensures pose stability before triggering)
- Debounce filtering (prevents duplicate bursts of characters)
"""

import time
from collections import deque, Counter
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

import config


class GestureRecognizer:
    """
    Filters raw frame-by-frame classifier predictions into stable, validated ISL gestures.
    """

    def __init__(self,
                 smoothing_window: int = config.SMOOTHING_WINDOW_SIZE,
                 stability_threshold: int = config.STABILITY_THRESHOLD,
                 debounce_seconds: float = config.DUPLICATE_DEBOUNCE_SECONDS,
                 confidence_threshold: float = config.PREDICTION_CONFIDENCE_THRESHOLD):

        self.smoothing_window = smoothing_window
        self.stability_threshold = stability_threshold
        self.debounce_seconds = debounce_seconds
        self.confidence_threshold = confidence_threshold

        # Rolling history buffer: stores tuples of (sign_name, confidence)
        self.history = deque(maxlen=smoothing_window)

        # State tracking
        self.current_stable_sign: Optional[str] = None
        self.current_confidence: float = 0.0
        self.stability_progress: float = 0.0

        # Debounce tracking
        self.last_emitted_sign: Optional[str] = None
        self.last_emit_time: float = 0.0

    @staticmethod
    def evaluate_lighting(frame_bgr: np.ndarray) -> Tuple[str, bool]:
        """
        Fast frame brightness calculation using cv2.mean.
        Takes < 0.05ms (zero lag).
        """
        # cv2.mean returns (mean_B, mean_G, mean_R)
        mean_bgr = cv2.mean(frame_bgr)[:3]
        mean_brightness = 0.299 * mean_bgr[2] + 0.587 * mean_bgr[1] + 0.114 * mean_bgr[0]

        if mean_brightness < 45.0:
            return "Poor Lighting (Dark)", False
        elif mean_brightness > 225.0:
            return "Overexposed Lighting", False
        return "Lighting OK", True

    def process(self, raw_sign: str, confidence: float) -> Dict[str, Any]:
        """
        Ingests a single raw frame prediction and returns stabilized gesture state.
        """
        self.history.append((raw_sign, confidence))

        if raw_sign == "No Hand":
            self.stability_progress = 0.0
            self.current_stable_sign = None
            return {
                "display_sign": "No Hand Detected",
                "confidence": 0.0,
                "is_stable": False,
                "stability_ratio": 0.0,
                "new_emission": None
            }

        valid_signs = [s for s, c in self.history if s != "No Hand"]
        if not valid_signs:
            self.stability_progress = 0.0
            return {
                "display_sign": "Detecting...",
                "confidence": confidence,
                "is_stable": False,
                "stability_ratio": 0.0,
                "new_emission": None
            }

        counts = Counter(valid_signs)
        most_common_sign, count = counts.most_common(1)[0]

        confs = [c for s, c in self.history if s == most_common_sign]
        avg_confidence = float(np.mean(confs)) if confs else confidence

        # Calculate stability progress ratio
        self.stability_progress = min(1.0, count / self.stability_threshold)
        self.current_confidence = avg_confidence

        # Validated if count meets stability threshold and average confidence >= threshold
        is_stable = (count >= self.stability_threshold) and (avg_confidence >= self.confidence_threshold)
        new_emission = None

        if is_stable:
            self.current_stable_sign = most_common_sign
            curr_time = time.time()
            time_since_last = curr_time - self.last_emit_time

            can_emit = (
                (most_common_sign != self.last_emitted_sign) or
                (time_since_last >= self.debounce_seconds)
            )

            if can_emit:
                new_emission = most_common_sign
                self.last_emitted_sign = most_common_sign
                self.last_emit_time = curr_time
        else:
            self.current_stable_sign = None

        display_text = most_common_sign

        return {
            "display_sign": display_text,
            "confidence": avg_confidence,
            "is_stable": is_stable,
            "stability_ratio": self.stability_progress,
            "new_emission": new_emission
        }

    def reset(self):
        """Resets the history buffer and debounce state."""
        self.history.clear()
        self.current_stable_sign = None
        self.last_emitted_sign = None
        self.last_emit_time = 0.0
        self.stability_progress = 0.0
