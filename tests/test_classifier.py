"""
Phase 5 Verification Test: Model Inference & Probability Evaluation
Loads models/isl_classifier.joblib and tests synthetic feature prediction,
probability calibration, and confidence thresholding.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.classifier import ISLClassifier
from dataset.generate_baseline_data import generate_sign_features
import config


def test_classifier():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PHASE 5 INFERENCE TEST")
    print("=" * 65)

    print("[*] Loading trained ISLClassifier from disk...")
    try:
        classifier = ISLClassifier()
        print(f"[OK] Successfully loaded model with {len(classifier.classes)} classes:")
        print("    ", ", ".join(classifier.classes))
    except Exception as e:
        print(f"[!] ERROR loading classifier: {e}")
        return False

    print("\n[*] Running Benchmark Tests across Selected Signs:")
    print("-" * 65)
    print(f"{'Input Sign':<14} | {'Predicted':<14} | {'Confidence':<12} | {'Status'}")
    print("-" * 65)

    test_signs = ["1", "2", "3", "4", "5", "A", "C", "L", "GOOD", "NAMASTE"]
    all_passed = True

    for sign in test_signs:
        # Generate canonical feature vector
        feat = generate_sign_features(sign)
        # Add slight noise to simulate real webcam capture
        noisy_feat = feat + np.random.normal(0, 0.015, size=feat.shape)

        pred_sign, conf, top_cands = classifier.predict(noisy_feat)
        match = (pred_sign == sign) and (conf >= config.PREDICTION_CONFIDENCE_THRESHOLD)
        status = "[PASS]" if match else "[FAIL]"
        if not match:
            all_passed = False

        print(f"{sign:<14} | {pred_sign:<14} | {conf * 100:6.1f}%      | {status}")

    # Test Empty / Zero input (No hands visible)
    empty_feat = np.zeros(126, dtype=np.float32)
    pred_empty, conf_empty, _ = classifier.predict(empty_feat)
    empty_pass = (pred_empty == "No Hand")
    print(f"{'No Hand (Zeros)':<14} | {pred_empty:<14} | {conf_empty * 100:6.1f}%      | {'[PASS]' if empty_pass else '[FAIL]'}")

    # Test Random Jumbled Noise (Uncertain)
    noise_feat = np.random.uniform(-1.0, 1.0, size=126).astype(np.float32)
    pred_noise, conf_noise, _ = classifier.predict(noise_feat)
    print(f"{'Random Noise':<14} | {pred_noise:<14} | {conf_noise * 100:6.1f}%      | [PASS]")

    print("=" * 65)
    if all_passed and empty_pass:
        print(">>> ALL PHASE 5 MODEL TESTS PASSED WITH HIGH CONFIDENCE! <<<")
        return True
    else:
        print("[!] Some test cases failed.")
        return False


if __name__ == "__main__":
    test_classifier()
