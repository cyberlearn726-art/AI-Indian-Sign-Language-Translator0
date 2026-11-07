"""
AI Model Training & Evaluation Engine for ISL Classifier
Trains a Multi-Class Classifier on 126-dimensional normalized landmark features.
Evaluates accuracy, precision, recall, and generates confusion metrics.
Saves model to models/isl_classifier.joblib and encoder to models/label_encoder.joblib.
"""

import sys
import csv
import json
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import config


def load_dataset():
    """Loads feature matrix X and label vector y from raw_landmarks.csv."""
    csv_path = config.RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run Phase 4 first.")

    X = []
    y = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            feats = [float(val) for val in row[:-1]]
            label = row[-1]
            X.append(feats)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def train_and_evaluate():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - MODEL TRAINING & EVALUATION")
    print("=" * 65)

    print("[*] Loading dataset...")
    X, y = load_dataset()
    print(f"[*] Loaded {len(X)} samples with {X.shape[1]} features across {len(np.unique(y))} classes.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Stratified Train/Test split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"[*] Training Samples: {len(X_train)} | Validation Samples: {len(X_test)}")

    # Initialize Random Forest Classifier
    # 120 trees provide high generalization, resistance to overfitting, and <1ms inference
    print("\n[*] Initializing Random Forest Classifier (120 Estimators, CPU Parallelized)...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    clf.fit(X_train, y_train)
    train_duration = time.time() - start_time
    print(f"[OK] Training completed in {train_duration:.2f} seconds.")

    # Validation Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[*] Overall Validation Accuracy: {accuracy * 100:.2f}%\n")

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=4, output_dict=False
    )
    print("Detailed Classification Report:")
    print("-" * 65)
    print(report)
    print("-" * 65)

    # Save trained artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    print(f"[OK] Saved model to: {config.MODEL_PATH.resolve()}")
    print(f"[OK] Saved label encoder to: {config.LABEL_ENCODER_PATH.resolve()}")

    # Save metrics summary
    metrics_summary = {
        "accuracy": float(accuracy),
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": list(label_encoder.classes_),
        "num_classes": int(num_classes),
        "train_duration_seconds": float(train_duration)
    }

    metrics_path = config.MODELS_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[OK] Saved metrics to: {metrics_path.name}")
    print("=" * 65)
    print(">>> MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY! <<<")
    print("=" * 65)
    return True


if __name__ == "__main__":
    train_and_evaluate()
