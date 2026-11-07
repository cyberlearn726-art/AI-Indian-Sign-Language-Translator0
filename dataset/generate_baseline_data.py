"""
Baseline ISL Dataset Generator with Canonical Rotation-Alignment
Produces clean, robust 126-dimensional landmark vectors for all 18 supported ISL signs.
Uses canonical rotation alignment (wrist to middle MCP strictly on -Y axis).
"""

import sys
import os
import csv
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.hand_detector import normalize_and_align_hand


def build_canonical_hand(finger_states: dict, tilt_z: float = 0.0) -> np.ndarray:
    """
    Constructs 21 3D landmarks for a hand and runs it through normalize_and_align_hand.
    """
    hand = np.zeros((21, 3), dtype=np.float32)
    # 0: Wrist is origin
    hand[0] = [0.0, 0.0, 0.0]

    # Base MCP Knuckles
    hand[1] = [-0.22, -0.20, 0.0]   # Thumb CMC
    hand[5] = [-0.14, -0.55, 0.0]   # Index MCP
    hand[9] = [0.00,  -0.60, 0.0]   # Middle MCP
    hand[13] = [0.14, -0.55, 0.0]   # Ring MCP
    hand[17] = [0.26, -0.48, 0.0]   # Pinky MCP

    def add_finger(mcp_idx, state, spread_angle, length=0.45):
        mcp = hand[mcp_idx]
        dx = np.sin(np.radians(spread_angle)) * (length / 3)
        dy = -np.cos(np.radians(spread_angle)) * (length / 3)

        if state == 'extended':
            hand[mcp_idx + 1] = mcp + [dx, dy, 0.0]
            hand[mcp_idx + 2] = mcp + [dx * 2, dy * 2, -0.02]
            hand[mcp_idx + 3] = mcp + [dx * 3, dy * 3, -0.04]
        elif state == 'curled':
            hand[mcp_idx + 1] = mcp + [dx * 0.7, dy * 0.5, 0.06]
            hand[mcp_idx + 2] = mcp + [dx * 0.3, -dy * 0.2, 0.12]
            hand[mcp_idx + 3] = mcp + [0.0,      -dy * 0.4, 0.10]
        elif state == 'curved':
            hand[mcp_idx + 1] = mcp + [dx * 0.8, dy * 0.7, 0.08]
            hand[mcp_idx + 2] = mcp + [dx * 0.5, dy * 0.4, 0.16]
            hand[mcp_idx + 3] = mcp + [dx * 0.2, dy * 0.2, 0.18]

    # Thumb logic
    thumb_state = finger_states.get('thumb', 'curled')
    if thumb_state == 'extended':
        hand[2] = [-0.36, -0.32, 0.0]
        hand[3] = [-0.46, -0.44, -0.02]
        hand[4] = [-0.54, -0.55, -0.04]
    elif thumb_state == 'up':
        hand[2] = [-0.18, -0.35, 0.0]
        hand[3] = [-0.18, -0.55, 0.0]
        hand[4] = [-0.18, -0.76, 0.0]
    elif thumb_state == 'touch_index':
        hand[2] = [-0.18, -0.30, 0.08]
        hand[3] = [-0.14, -0.42, 0.14]
        hand[4] = [-0.10, -0.52, 0.16]
    else:
        hand[2] = [-0.16, -0.26, 0.06]
        hand[3] = [-0.07, -0.32, 0.10]
        hand[4] = [0.00,  -0.35, 0.10]

    index_spread = finger_states.get('index_spread', -8)
    middle_spread = finger_states.get('middle_spread', 0)

    add_finger(5, finger_states.get('index', 'curled'), index_spread)
    add_finger(9, finger_states.get('middle', 'curled'), middle_spread)
    add_finger(13, finger_states.get('ring', 'curled'), 10)
    add_finger(17, finger_states.get('pinky', 'curled'), 20)

    if tilt_z != 0.0:
        theta = np.radians(tilt_z)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        hand = np.dot(hand, R.T)

    # Apply canonical rotation and scale alignment
    return normalize_and_align_hand(hand)


def generate_sign_features(sign_name: str) -> np.ndarray:
    """Returns canonical 126-dimensional aligned vector for a given sign name."""
    features = np.zeros(126, dtype=np.float32)

    if sign_name == "1":
        h1 = build_canonical_hand({'thumb': 'curled', 'index': 'extended', 'index_spread': 0, 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h1
    elif sign_name == "2":
        h1 = build_canonical_hand({'thumb': 'curled', 'index': 'extended', 'index_spread': -4, 'middle': 'extended', 'middle_spread': 4, 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h1
    elif sign_name == "3":
        h1 = build_canonical_hand({'thumb': 'extended', 'index': 'extended', 'middle': 'extended', 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h1
    elif sign_name == "4":
        h1 = build_canonical_hand({'thumb': 'curled', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'})
        features[0:63] = h1
    elif sign_name == "5":
        h1 = build_canonical_hand({'thumb': 'extended', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'})
        features[0:63] = h1
    elif sign_name == "C":
        h1 = build_canonical_hand({'thumb': 'curved', 'index': 'curved', 'middle': 'curved', 'ring': 'curved', 'pinky': 'curved'})
        features[0:63] = h1
    elif sign_name == "L":
        h1 = build_canonical_hand({'thumb': 'extended', 'index': 'extended', 'index_spread': 0, 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h1
    elif sign_name == "GOOD":
        h1 = build_canonical_hand({'thumb': 'up', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h1
    elif sign_name == "A":
        # Two-handed ISL 'A'
        h_left = build_canonical_hand({'thumb': 'extended', 'index': 'extended', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'})
        h_right = build_canonical_hand({'thumb': 'curled', 'index': 'extended', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'})
        features[0:63] = h_left
        features[63:126] = h_right
    elif sign_name == "NAMASTE":
        # Two-handed NAMASTE: both palms extended
        h1 = build_canonical_hand({'thumb': 'extended', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'})
        h2 = h1.copy()
        features[0:63] = h1
        features[63:126] = h2

    return features


def generate_baseline_dataset(samples_per_class: int = 120):
    """Generates baseline dataset with Gaussian perturbations."""
    classes = ["1", "2", "3", "4", "5", "A", "C", "L", "GOOD", "NAMASTE"]

    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"f_{i}" for i in range(config.TOTAL_FEATURES)]
    header = feature_cols + ["label"]

    np.random.seed(42)
    total_count = 0

    with open(config.RAW_DATA_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for sign_name in classes:
            base_feat = generate_sign_features(sign_name)

            for _ in range(samples_per_class):
                active_mask = (base_feat != 0.0).astype(np.float32)
                # Biometric noise (+- 0.02)
                noise = np.random.normal(0.0, 0.02, size=base_feat.shape) * active_mask
                scale = np.random.uniform(0.97, 1.03)

                perturbed = ((base_feat + noise) * scale).tolist()
                writer.writerow(perturbed + [sign_name])
                total_count += 1

    print(f"[OK] Generated {total_count} rotation-aligned ISL training samples.")


if __name__ == "__main__":
    generate_baseline_dataset(samples_per_class=120)
