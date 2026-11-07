"""
Phase 2 Verification Test: OpenCV Webcam Integration
Tests camera capture, multi-threading, FPS metering, and GUI rendering.
Press 'q' or 'ESC' to exit the test.
"""

import sys
import time
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraStream
import config


def run_camera_test():
    print("=" * 60)
    print("AI INDIAN SIGN LANGUAGE TRANSLATOR - PHASE 2 CAMERA TEST")
    print("=" * 60)
    print(f"[*] Attempting to open Camera Index: {config.CAMERA_INDEX}")
    print(f"[*] Target Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print("[*] Controls: Press 'q' or 'ESC' on the video window to quit.")
    print("=" * 60)

    camera = CameraStream(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        target_fps=config.TARGET_FPS
    )

    success = camera.start()
    if not success:
        print("\n[!] ERROR: Could not open the webcam.")
        print("    Troubleshooting steps:")
        print("    1. If you are using an external USB webcam, change CAMERA_INDEX to 1 in config.py.")
        print("    2. Check Windows Settings -> Privacy & Security -> Camera -> Allow desktop apps to access camera.")
        print("    3. Ensure no other application (Zoom, Teams, Skype, Browser) is currently using the webcam.")
        return False

    window_name = "ISL Translator - Phase 2 Webcam Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                # Small pause while camera warms up
                time.sleep(0.01)
                continue

            # Calculate loop FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # Mirror frame horizontally for intuitive interaction
            display_frame = cv2.flip(frame, 1)

            # Draw status HUD overlay
            h, w = display_frame.shape[:2]
            
            # Semi-transparent top bar
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            # Render text information
            cv2.putText(display_frame, "AI ISL Translator - Phase 2: Camera Feed OK",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

            hud_info = f"FPS: {fps_display:.1f} | Res: {w}x{h} | Press 'q' to Exit"
            cv2.putText(display_frame, hud_info,
                        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[*] Exit key pressed by user.")
                break

    finally:
        print("[*] Cleaning up and releasing camera...")
        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Phase 2 Camera Test completed successfully.")

    return True


if __name__ == "__main__":
    run_camera_test()
