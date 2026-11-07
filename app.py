"""
AI-Powered Indian Sign Language Translator
Main Application Entry Point (Desktop PySide6 Application)
"""

import sys
import os
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from gui.main_window import ISLMainWindow


def main():
    # Enable High-DPI display attributes for sharp rendering on 1080p/4K Windows laptops
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AI Indian Sign Language Translator")
    app.setOrganizationName("IndianSignLanguageAI")

    window = ISLMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
