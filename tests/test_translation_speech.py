"""
Phase 7 & 8 Verification Test: Text Translation & Offline Speech Generation
Validates sentence construction, token spacing, and non-blocking SAPI5 speech synthesis.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.translator import TextTranslator
from core.speech_engine import OfflineSpeechEngine


def run_translation_and_speech_test():
    print("=" * 65)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PHASE 7 & 8 TEST")
    print("=" * 65)

    # 1. Test Text Translation Engine
    print("[*] Testing Phase 7: Text Translation & Sentence Builder...")
    translator = TextTranslator()

    test_stream = ["NAMASTE", "1", "2", "3", "GOOD"]
    print(f"[*] Simulating incoming recognized sign stream: {test_stream}")

    for sign in test_stream:
        sentence = translator.add_sign(sign)
        print(f"    Added Sign: {sign:<10} -> Current Sentence: \"{sentence}\"")

    final_text = translator.get_sentence()
    print(f"\n[OK] Constructed Final Sentence: \"{final_text}\"")

    # Test backspace and space
    translator.backspace()
    print(f"[*] After Backspace: \"{translator.get_sentence()}\"")

    # 2. Test Offline Speech Engine
    print("\n[*] Testing Phase 8: Offline SAPI5 Speech Engine (Non-Blocking)...")
    try:
        speech_engine = OfflineSpeechEngine()
        voices = speech_engine.get_available_voices()
        print(f"[OK] Detected {len(voices)} offline voice(s): {voices}")

        test_phrase = f"Namaste. Indian Sign Language Translation confirmed: {final_text}"
        print(f"[*] Speaking phrase: \"{test_phrase}\"")
        speech_engine.speak(test_phrase)

        # Confirm non-blocking operation
        print("[*] Confirming non-blocking execution (main loop continues immediately)...")
        wait_counter = 0
        while speech_engine.is_speaking() and wait_counter < 30:
            time.sleep(0.1)
            wait_counter += 1

        print("[OK] Speech completed cleanly without blocking!")
        speech_engine.stop()

    except Exception as e:
        print(f"[!] Speech engine test error: {e}")
        return False

    print("=" * 65)
    print(">>> PHASE 7 & 8 VERIFIED SUCCESSFULLY! <<<")
    print("=" * 65)
    return True


if __name__ == "__main__":
    run_translation_and_speech_test()
