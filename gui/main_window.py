"""
Main Window Desktop Application Interface using PySide6
Includes all 12 required components + Translation History Log + Rock-Solid SAPI5 Speech.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QProgressBar,
    QFrame, QMessageBox, QListWidget, QListWidgetItem, QSizePolicy
)
import cv2
import numpy as np

import config
from core.camera import CameraStream
from core.hand_detector import HandDetector
from core.classifier import ISLClassifier
from core.recognizer import GestureRecognizer
from core.translator import TextTranslator
from core.speech_engine import OfflineSpeechEngine
from gui.styles import DARK_THEME_QSS


class VisionWorkerThread(QThread):
    """
    Dedicated background worker thread for camera capture, landmark extraction,
    and AI inference. Emits signals to update the PySide6 UI at 30 FPS.
    """
    frame_ready = Signal(QImage)
    prediction_ready = Signal(str, float, float)     # sign_name, confidence, stability_ratio
    sign_confirmed = Signal(str)                     # confirmed sign to append
    telemetry_ready = Signal(float, int, str)        # fps, hands_count, lighting_msg
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.camera = None
        self.detector = None
        self.classifier = None
        self.recognizer = None

    def run(self):
        self.is_running = True

        self.camera = CameraStream()
        if not self.camera.start():
            self.error_occurred.emit("Failed to open webcam. Check device index and permissions.")
            return

        try:
            self.detector = HandDetector()
            self.classifier = ISLClassifier()
            self.recognizer = GestureRecognizer()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load AI models: {e}")
            self.camera.stop()
            return

        prev_time = time.time()
        fps_display = 0.0
        fail_count = 0

        while self.is_running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                fail_count += 1
                if fail_count > 50:
                    self.error_occurred.emit("Webcam opened, but is not returning video frames.\nPlease ensure another app (Zoom, Teams, Chrome) is not using the camera.")
                    break
                self.msleep(20)
                continue
            fail_count = 0

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # Mirror frame horizontally for intuitive display
            mirrored = cv2.flip(frame, 1)

            # Check lighting
            lighting_msg, _ = self.recognizer.evaluate_lighting(mirrored)

            # Extract hand landmarks with deterministic spatial sorting
            detection_res, annotated, features = self.detector.process_frame(mirrored)
            num_hands = len(detection_res.hand_landmarks) if detection_res.hand_landmarks else 0

            # Predict sign
            raw_sign, raw_conf, candidates = self.classifier.predict(features)

            # Temporal smoothing and debounce
            state = self.recognizer.process(raw_sign, raw_conf)

            # Emit predictions
            self.prediction_ready.emit(
                state["display_sign"], state["confidence"], state["stability_ratio"]
            )

            if state["new_emission"]:
                self.sign_confirmed.emit(state["new_emission"])

            self.telemetry_ready.emit(fps_display, num_hands, lighting_msg)

            # Scale frame in worker thread using fast SIMD cv2.resize to offload GUI thread
            h, w, ch = annotated.shape
            if w != 640 or h != 480:
                render_frame = cv2.resize(annotated, (640, 480), interpolation=cv2.INTER_LINEAR)
            else:
                render_frame = annotated

            rgb_frame = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
            rh, rw, rch = rgb_frame.shape
            bytes_per_line = rch * rw
            q_img = QImage(rgb_frame.data, rw, rh, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(q_img.copy())

            self.msleep(1)

        # Cleanup
        if self.detector:
            self.detector.close()
        if self.camera:
            self.camera.stop()

    def stop(self):
        self.is_running = False
        self.wait(1500)


class ISLMainWindow(QMainWindow):
    """
    Main PySide6 Desktop Application Window for Indian Sign Language Translation.
    Includes translation history log, editing tools, and native speech engine.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Indian Sign Language Translator")
        self.resize(1240, 780)
        self.setMinimumSize(1020, 680)

        self.setStyleSheet(DARK_THEME_QSS)

        # Core engines
        self.translator = TextTranslator()
        self.speech_engine = OfflineSpeechEngine()

        self.worker: Optional[VisionWorkerThread] = None

        self._build_ui()

    def _build_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # =========================================================================
        # 1. HEADER SECTION
        # =========================================================================
        header_layout = QHBoxLayout()

        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        self.title_label = QLabel("AI INDIAN SIGN LANGUAGE TRANSLATOR")
        self.title_label.setObjectName("TitleLabel")
        self.subtitle_label = QLabel("Real-Time Bimanual Computer Vision Translation System (100% Offline)")
        self.subtitle_label.setObjectName("SubtitleLabel")
        title_container.addWidget(self.title_label)
        title_container.addWidget(self.subtitle_label)

        header_layout.addLayout(title_container)
        header_layout.addStretch()

        self.status_badge = QLabel("● SYSTEM IDLE")
        self.status_badge.setObjectName("StatusBadgeInactive")
        header_layout.addWidget(self.status_badge)

        main_layout.addLayout(header_layout)

        # =========================================================================
        # 2. BODY SPLIT (Left: Video | Right: Translation, History & Controls)
        # =========================================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        # --- LEFT PANE: VIDEO STREAM CARD ---
        video_card = QFrame()
        video_card.setProperty("class", "CardFrame")
        video_layout = QVBoxLayout(video_card)
        video_layout.setSpacing(8)

        video_header = QLabel("LIVE CAMERA FEED & HAND TRACKING")
        video_header.setProperty("class", "CardHeader")
        video_layout.addWidget(video_header)

        self.video_label = QLabel()
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(580, 435)
        self.video_label.setScaledContents(True)  # Hardware-accelerated scaling, zero CPU lag
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setText("Camera is currently stopped.\nClick 'Start Camera' below to begin translation.")
        video_layout.addWidget(self.video_label)

        # Telemetry bar
        telemetry_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.hands_label = QLabel("Hands: 0/2")
        self.lighting_label = QLabel("Lighting: --")
        for lbl in (self.fps_label, self.hands_label, self.lighting_label):
            lbl.setStyleSheet("color: #8E9297; font-weight: 500;")
            telemetry_layout.addWidget(lbl)
        telemetry_layout.addStretch()
        video_layout.addLayout(telemetry_layout)

        # Camera buttons
        cam_buttons_layout = QHBoxLayout()
        self.btn_start_cam = QPushButton("▶ Start Camera")
        self.btn_start_cam.setObjectName("PrimaryButton")
        self.btn_start_cam.clicked.connect(self.start_camera)

        self.btn_stop_cam = QPushButton("⏹ Stop Camera")
        self.btn_stop_cam.setObjectName("DangerButton")
        self.btn_stop_cam.setEnabled(False)
        self.btn_stop_cam.clicked.connect(self.stop_camera)

        cam_buttons_layout.addWidget(self.btn_start_cam)
        cam_buttons_layout.addWidget(self.btn_stop_cam)
        video_layout.addLayout(cam_buttons_layout)

        body_layout.addWidget(video_card, stretch=6)

        # --- RIGHT PANE: TRANSLATION DASHBOARD ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 2.1 Recognized Sign Card
        recog_card = QFrame()
        recog_card.setProperty("class", "CardFrame")
        recog_layout = QVBoxLayout(recog_card)
        recog_layout.setSpacing(6)

        recog_header = QLabel("ACTIVE RECOGNITION")
        recog_header.setProperty("class", "CardHeader")
        recog_layout.addWidget(recog_header)

        self.sign_display = QLabel("NO HAND")
        self.sign_display.setObjectName("SignDisplayLabel")
        recog_layout.addWidget(self.sign_display)

        # Confidence & Stability meters
        meters_layout = QHBoxLayout()

        conf_box = QVBoxLayout()
        conf_header = QHBoxLayout()
        conf_header.addWidget(QLabel("Confidence:"))
        self.conf_percent_label = QLabel("0%")
        self.conf_percent_label.setStyleSheet("color: #00E676; font-weight: 700;")
        conf_header.addWidget(self.conf_percent_label)
        conf_box.addLayout(conf_header)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        conf_box.addWidget(self.conf_bar)
        meters_layout.addLayout(conf_box)

        stab_box = QVBoxLayout()
        stab_header = QHBoxLayout()
        stab_header.addWidget(QLabel("Hold Stability:"))
        self.stab_percent_label = QLabel("0%")
        self.stab_percent_label.setStyleSheet("color: #FFB300; font-weight: 700;")
        stab_header.addWidget(self.stab_percent_label)
        stab_box.addLayout(stab_header)
        self.stab_bar = QProgressBar()
        self.stab_bar.setObjectName("StabilityBar")
        self.stab_bar.setRange(0, 100)
        self.stab_bar.setValue(0)
        stab_box.addWidget(self.stab_bar)
        meters_layout.addLayout(stab_box)

        recog_layout.addLayout(meters_layout)
        right_panel.addWidget(recog_card)

        # 2.2 Translation Text Box Card
        trans_card = QFrame()
        trans_card.setProperty("class", "CardFrame")
        trans_layout = QVBoxLayout(trans_card)
        trans_layout.setSpacing(6)

        trans_header = QLabel("CURRENT TRANSLATION")
        trans_header.setProperty("class", "CardHeader")
        trans_layout.addWidget(trans_header)

        self.text_box = QTextEdit()
        self.text_box.setObjectName("TranslatedTextBox")
        self.text_box.setPlaceholderText("Recognized signs will construct a sentence here...")
        self.text_box.setMaximumHeight(85)
        trans_layout.addWidget(self.text_box)

        # Actions Row
        actions_grid = QHBoxLayout()
        self.btn_space = QPushButton("␣ Space")
        self.btn_space.clicked.connect(self.on_space_clicked)

        self.btn_backspace = QPushButton("⌫ Backspace")
        self.btn_backspace.clicked.connect(self.on_backspace_clicked)

        self.btn_clear = QPushButton("🗑 Clear")
        self.btn_clear.clicked.connect(self.on_clear_clicked)

        actions_grid.addWidget(self.btn_space)
        actions_grid.addWidget(self.btn_backspace)
        actions_grid.addWidget(self.btn_clear)
        trans_layout.addLayout(actions_grid)

        # Speak Button
        self.btn_speak = QPushButton("🔊 Speak Translation")
        self.btn_speak.setObjectName("SuccessButton")
        self.btn_speak.clicked.connect(self.on_speak_clicked)
        trans_layout.addWidget(self.btn_speak)

        right_panel.addWidget(trans_card)

        # 2.3 NEW: TRANSLATION HISTORY CARD (Requested Feature)
        history_card = QFrame()
        history_card.setProperty("class", "CardFrame")
        hist_layout = QVBoxLayout(history_card)
        hist_layout.setSpacing(6)

        hist_header_layout = QHBoxLayout()
        hist_header = QLabel("📜 TRANSLATION HISTORY")
        hist_header.setProperty("class", "CardHeader")
        hist_header_layout.addWidget(hist_header)
        hist_header_layout.addStretch()

        self.btn_speak_selected = QPushButton("🔊 Speak Selected")
        self.btn_speak_selected.clicked.connect(self.on_speak_selected_clicked)
        self.btn_copy_hist = QPushButton("📋 Copy")
        self.btn_copy_hist.clicked.connect(self.on_copy_history_clicked)
        self.btn_clear_hist = QPushButton("🗑 Clear History")
        self.btn_clear_hist.clicked.connect(self.on_clear_history_clicked)

        hist_header_layout.addWidget(self.btn_speak_selected)
        hist_header_layout.addWidget(self.btn_copy_hist)
        hist_header_layout.addWidget(self.btn_clear_hist)
        hist_layout.addLayout(hist_header_layout)

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        self.history_list.setMaximumHeight(140)
        self.history_list.itemDoubleClicked.connect(self.on_history_double_clicked)
        hist_layout.addWidget(self.history_list)

        right_panel.addWidget(history_card)

        # Bottom Utilities Row
        bottom_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("⚡ Calibrate Signs")
        self.btn_calibrate.clicked.connect(self.on_calibrate_clicked)

        self.btn_exit = QPushButton("✕ Exit Application")
        self.btn_exit.setObjectName("DangerButton")
        self.btn_exit.clicked.connect(self.close)

        bottom_row.addWidget(self.btn_calibrate)
        bottom_row.addStretch()
        bottom_row.addWidget(self.btn_exit)
        right_panel.addLayout(bottom_row)

        body_layout.addLayout(right_panel, stretch=5)
        main_layout.addLayout(body_layout)

        self.statusBar().showMessage("Ready | 100% Offline Mode | Native SAPI5 Speech Active")

    # =========================================================================
    # SLOTS & CONTROLS
    # =========================================================================
    def start_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)

        self.status_badge.setText("● RECOGNIZING")
        self.status_badge.setObjectName("StatusBadgeActive")
        self.status_badge.setStyle(self.status_badge.style())

        self.worker = VisionWorkerThread()
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.prediction_ready.connect(self.update_prediction_ui)
        self.worker.sign_confirmed.connect(self.on_sign_confirmed)
        self.worker.telemetry_ready.connect(self.update_telemetry)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.start()

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)

        self.status_badge.setText("● SYSTEM IDLE")
        self.status_badge.setObjectName("StatusBadgeInactive")
        self.status_badge.setStyle(self.status_badge.style())

        self.video_label.setText("Camera is currently stopped.\nClick 'Start Camera' below to begin translation.")
        self.sign_display.setText("NO HAND")
        self.conf_bar.setValue(0)
        self.stab_bar.setValue(0)
        self.fps_label.setText("FPS: --")

    @Slot(QImage)
    def update_video_frame(self, q_img: QImage):
        # Direct zero-copy setPixmap - no CPU rescaling lag
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    @Slot(str, float, float)
    def update_prediction_ui(self, sign_name: str, confidence: float, stability_ratio: float):
        self.sign_display.setText(sign_name)

        conf_pct = int(confidence * 100)
        self.conf_bar.setValue(conf_pct)
        self.conf_percent_label.setText(f"{conf_pct}%")

        stab_pct = int(stability_ratio * 100)
        self.stab_bar.setValue(stab_pct)
        self.stab_percent_label.setText(f"{stab_pct}%")

    @Slot(str)
    def on_sign_confirmed(self, sign: str):
        new_sentence = self.translator.add_sign(sign)
        self.text_box.setPlainText(new_sentence)
        self.statusBar().showMessage(f"Confirmed Sign: {sign} | Sentence Updated", 2500)

    @Slot(float, int, str)
    def update_telemetry(self, fps: float, num_hands: int, lighting_msg: str):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.hands_label.setText(f"Hands: {num_hands}/2")
        self.lighting_label.setText(lighting_msg)

    @Slot(str)
    def on_worker_error(self, err_msg: str):
        self.stop_camera()
        QMessageBox.critical(self, "Camera / AI Error", err_msg)

    def on_space_clicked(self):
        new_text = self.translator.add_space()
        self.text_box.setPlainText(new_text)

    def on_backspace_clicked(self):
        new_text = self.translator.backspace()
        self.text_box.setPlainText(new_text)

    def on_clear_clicked(self):
        current_text = self.text_box.toPlainText().strip()
        if current_text:
            self._archive_to_history(current_text)

        self.translator.clear()
        self.text_box.clear()
        self.statusBar().showMessage("Translation cleared and saved to History.", 2500)

    def on_speak_clicked(self):
        text_to_speak = self.text_box.toPlainText().strip()
        if not text_to_speak:
            text_to_speak = self.translator.get_sentence()

        if text_to_speak:
            self.statusBar().showMessage(f"Speaking: '{text_to_speak}'...", 3000)
            self.speech_engine.speak(text_to_speak)
            self._archive_to_history(text_to_speak)
        else:
            self.statusBar().showMessage("No text to speak.", 2000)

    def _archive_to_history(self, text: str):
        """Adds a finalized sentence to the history log with timestamp."""
        if not text or not text.strip():
            return
        # Prevent consecutive duplicates in history
        if self.history_list.count() > 0:
            last_item = self.history_list.item(0).text()
            if text in last_item:
                return

        time_str = datetime.now().strftime("%H:%M:%S")
        entry_text = f"[{time_str}] {text.strip()}"

        item = QListWidgetItem(entry_text)
        # Insert at top (most recent first)
        self.history_list.insertItem(0, item)

    def on_speak_selected_clicked(self):
        selected_item = self.history_list.currentItem()
        if selected_item:
            # Strip timestamp "[HH:MM:SS] "
            full_text = selected_item.text()
            clean_text = full_text.split("] ", 1)[-1] if "] " in full_text else full_text
            self.speech_engine.speak(clean_text)
            self.statusBar().showMessage(f"Re-speaking: '{clean_text}'", 2500)

    def on_history_double_clicked(self, item: QListWidgetItem):
        self.on_speak_selected_clicked()

    def on_copy_history_clicked(self):
        items = [self.history_list.item(i).text() for i in range(self.history_list.count())]
        if items:
            full_history = "\n".join(items)
            clipboard: QClipboard = QGuiApplication.clipboard()
            clipboard.setText(full_history)
            self.statusBar().showMessage("History copied to clipboard!", 2500)

    def on_clear_history_clicked(self):
        self.history_list.clear()
        self.statusBar().showMessage("History cleared.", 2000)

    def on_calibrate_clicked(self):
        msg = (
            "To calibrate or add new signs to the AI model:\n\n"
            "1. Close this main app window.\n"
            "2. Run 'python quick_train.py' in your terminal.\n"
            "3. Follow the 1-minute wizard to record your hand poses.\n"
            "4. Relaunch the application."
        )
        QMessageBox.information(self, "Quick Calibration Wizard", msg)

    def closeEvent(self, event):
        self.stop_camera()
        self.speech_engine.stop()
        event.accept()
