"""
ISL Runtime Classifier Engine
Loads trained model and label encoder to perform low-latency CPU inference.
Applies calibrated probability scoring and top candidate extraction.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import joblib

import config


class ISLClassifier:
    """
    Offline inference engine for Indian Sign Language gesture recognition.
    """

    def __init__(self,
                 model_path: Optional[Path] = None,
                 encoder_path: Optional[Path] = None,
                 confidence_threshold: float = config.PREDICTION_CONFIDENCE_THRESHOLD):

        if model_path is None:
            model_path = config.MODEL_PATH
        if encoder_path is None:
            encoder_path = config.LABEL_ENCODER_PATH

        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(
                f"Trained model or label encoder not found.\n"
                f"Model: {model_path}\nEncoder: {encoder_path}\n"
                f"Run 'python models/train_model.py' to train the model first."
            )

        self.confidence_threshold = confidence_threshold
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.classes = list(self.label_encoder.classes_)

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on a 126-dimensional normalized landmark vector.

        Returns:
            - predicted_sign: str (Highest probability class name, or "No Hand")
            - confidence: float (0.0 to 1.0)
            - top_candidates: dict mapping top 3 candidate labels to probabilities
        """
        # 1. Check if hands are present
        if np.all(feature_vector == 0.0) or np.max(np.abs(feature_vector)) < 1e-3:
            return "No Hand", 0.0, {}

        # 2. Reshape vector for scikit-learn (1, 126)
        x_input = feature_vector.reshape(1, -1)

        # 3. Predict class probabilities
        probabilities = self.model.predict_proba(x_input)[0]

        # 4. Find highest probability class
        best_idx = int(np.argmax(probabilities))
        top_confidence = float(probabilities[best_idx])
        top_sign = str(self.label_encoder.inverse_transform([best_idx])[0])

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_candidates = {
            str(self.label_encoder.classes_[idx]): float(probabilities[idx])
            for idx in top_indices
        }

        return top_sign, top_confidence, top_candidates
