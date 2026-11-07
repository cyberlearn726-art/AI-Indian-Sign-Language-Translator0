"""
Modern Dark Theme QSS Stylesheet for AI ISL Translator
Exhibition-grade, high-contrast dark UI palette with vibrant accent states.
"""

DARK_THEME_QSS = """
/* Global Window Styling */
QMainWindow, QDialog {
    background-color: #121316;
    color: #E4E6EB;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

QWidget {
    font-size: 13px;
    color: #E4E6EB;
}

/* Header Styling */
QLabel#TitleLabel {
    font-size: 22px;
    font-weight: 700;
    color: #00E676;
    letter-spacing: 0.5px;
}

QLabel#SubtitleLabel {
    font-size: 12px;
    color: #B0B3B8;
    margin-top: -2px;
}

/* Status Badges */
QLabel#StatusBadgeActive {
    background-color: rgba(0, 230, 118, 0.15);
    color: #00E676;
    border: 1px solid #00E676;
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 12px;
}

QLabel#StatusBadgeInactive {
    background-color: rgba(255, 82, 82, 0.15);
    color: #FF5252;
    border: 1px solid #FF5252;
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 12px;
}

/* Section Cards / Panels */
QFrame.CardFrame {
    background-color: #1E1F24;
    border: 1px solid #2E3038;
    border-radius: 12px;
    padding: 12px;
}

QLabel.CardHeader {
    font-size: 13px;
    font-weight: 700;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

/* Video Canvas */
QLabel#VideoLabel {
    background-color: #0A0A0C;
    border: 2px solid #2E3038;
    border-radius: 12px;
}

/* Large Recognized Sign Typography */
QLabel#SignDisplayLabel {
    font-size: 28px;
    font-weight: 800;
    color: #00E676;
    background-color: #16171B;
    border: 1px solid #2E3038;
    border-radius: 10px;
    padding: 10px;
    qproperty-alignment: AlignCenter;
}

/* Translated Text Box */
QTextEdit#TranslatedTextBox {
    background-color: #16171B;
    border: 1px solid #2E3038;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
    color: #FFFFFF;
    line-height: 1.4;
}

QTextEdit#TranslatedTextBox:focus {
    border: 1px solid #2979FF;
}

/* History List Widget */
QListWidget#HistoryList {
    background-color: #16171B;
    border: 1px solid #2E3038;
    border-radius: 10px;
    padding: 6px;
    font-size: 13px;
    color: #E4E6EB;
}

QListWidget#HistoryList::item {
    background-color: #24252B;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 4px;
    color: #FFFFFF;
}

QListWidget#HistoryList::item:hover {
    background-color: #2F3138;
}

QListWidget#HistoryList::item:selected {
    background-color: #2979FF;
    color: #FFFFFF;
    font-weight: 600;
}

/* Progress Bars */
QProgressBar {
    background-color: #16171B;
    border: 1px solid #2E3038;
    border-radius: 6px;
    height: 16px;
    text-align: center;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E676, stop:1 #00B0FF);
    border-radius: 5px;
}

QProgressBar#StabilityBar::chunk {
    background-color: #FFB300;
    border-radius: 5px;
}

/* Push Buttons */
QPushButton {
    background-color: #2E3038;
    color: #FFFFFF;
    border: 1px solid #3E404C;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 12px;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #3E404C;
    border-color: #555768;
}

QPushButton:pressed {
    background-color: #24252B;
}

/* Primary Action Buttons */
QPushButton#PrimaryButton {
    background-color: #2979FF;
    border-color: #2979FF;
    color: #FFFFFF;
}

QPushButton#PrimaryButton:hover {
    background-color: #448AFF;
}

QPushButton:disabled {
    background-color: #202226;
    color: #606368;
    border-color: #2E3038;
}

QPushButton#SuccessButton {
    background-color: #00C853;
    border-color: #00C853;
    color: #000000;
    font-weight: 700;
    font-size: 13px;
    padding: 9px 16px;
}

QPushButton#SuccessButton:hover {
    background-color: #00E676;
}

QPushButton#DangerButton {
    background-color: rgba(255, 82, 82, 0.2);
    border: 1px solid #FF5252;
    color: #FF5252;
}

QPushButton#DangerButton:hover {
    background-color: #FF5252;
    color: #FFFFFF;
}

/* Status bar */
QStatusBar {
    background-color: #0E0F12;
    color: #B0B3B8;
    border-top: 1px solid #1E1F24;
}
"""
