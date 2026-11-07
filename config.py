"""
Central Configuration for AI Indian Sign Language Translator
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
ASSETS_DIR = BASE_DIR / "assets"

# Camera Configuration
CAMERA_INDEX = 0             # 0 for default built-in webcam, 1 for external USB webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# MediaPipe Hands Configuration
MAX_NUM_HANDS = 2            # 2 hands required for authentic ISL
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.50

# Landmark Vector Dimensions
# 2 hands * 21 landmarks * 3 coordinates (x, y, z) = 126 normalized features
NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS_PER_LANDMARK = 3
TOTAL_FEATURES = MAX_NUM_HANDS * NUM_LANDMARKS_PER_HAND * NUM_COORDS_PER_LANDMARK  # 126

# Model & Recognition Thresholds
PREDICTION_CONFIDENCE_THRESHOLD = 0.50  # Lowered from 0.80 for responsive human gesture detection
SMOOTHING_WINDOW_SIZE = 10            # Rolling window frames for consensus
STABILITY_THRESHOLD = 6               # 6 consecutive frames of consensus (~0.2s) to lock in
DUPLICATE_DEBOUNCE_SECONDS = 1.0      # Prevent repeating same character too rapidly

# Model File Paths
MODEL_PATH = MODELS_DIR / "isl_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
RAW_DATA_PATH = DATASET_DIR / "raw_landmarks.csv"
CLASSES_CONFIG_PATH = DATASET_DIR / "sign_classes.json"

# Text-To-Speech Configuration
TTS_RATE = 150                        # Words per minute
TTS_VOLUME = 1.0                      # 0.0 to 1.0
