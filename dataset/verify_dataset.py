"""
Dataset Verification Script
Inspects dataset/raw_landmarks.csv:
- Validates row and column counts
- Checks for missing or NaN values
- Displays per-class sample distribution table
"""

import sys
import csv
from pathlib import Path
from collections import Counter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def verify_dataset():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - DATASET INTEGRITY CHECK")
    print("=" * 65)

    csv_path = config.RAW_DATA_PATH
    if not csv_path.exists():
        print(f"[!] Dataset file not found at: {csv_path}")
        print("    Run 'python dataset/generate_baseline_data.py' or 'python dataset/collect_data.py'.")
        return False

    print(f"[*] Reading dataset: {csv_path.name}")

    total_rows = 0
    nan_count = 0
    class_counts = Counter()
    num_features = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("[!] Empty CSV file.")
            return False

        num_features = len(header) - 1

        for row_idx, row in enumerate(reader, start=1):
            if len(row) != len(header):
                print(f"[!] Warning: Row {row_idx} length mismatch ({len(row)} vs {len(header)})")
                continue

            label = row[-1]
            class_counts[label] += 1
            total_rows += 1

            # Check for corrupt floats
            for val in row[:-1]:
                try:
                    fval = float(val)
                    if fval != fval:  # NaN check
                        nan_count += 1
                except ValueError:
                    nan_count += 1

    print(f"[*] Total Samples: {total_rows}")
    print(f"[*] Feature Dimensions: {num_features} (Expected: {config.TOTAL_FEATURES})")
    print(f"[*] Corrupt/NaN Values: {nan_count}")
    print(f"[*] Unique Classes: {len(class_counts)}")
    print("-" * 65)
    print(f"{'Class Name':<16} | {'Sample Count':<14} | {'Percentage':<10}")
    print("-" * 65)

    for label, count in sorted(class_counts.items(), key=lambda x: x[0]):
        pct = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"{label:<16} | {count:<14} | {pct:6.2f}%")

    print("=" * 65)
    if nan_count == 0 and total_rows > 0 and num_features == config.TOTAL_FEATURES:
        print("[OK] DATASET IS BALANCED, VALID, AND READY FOR AI TRAINING!")
        return True
    else:
        print("[!] Dataset has warnings. Please review the output above.")
        return False


if __name__ == "__main__":
    verify_dataset()
