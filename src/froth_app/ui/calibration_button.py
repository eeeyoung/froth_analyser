"""
calibration_button.py — Split dropdown "Calibration" button.

A QWidget containing:
    - A main QPushButton (stretches to fill)
    - A narrow arrow QPushButton (fixed 22 px) that opens a QMenu

States
------
    idle    : grey background, label "Calibration"
    active  : red background, label "✓ Confirm Overflow"

Signals
-------
    overflow_requested  : user chose "Overflow Direction" from the menu
    overflow_confirmed  : user clicked the main button while active
    ruler_requested     : placeholder (not yet functional)
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QMenu
from PySide6.QtGui import QAction, QCursor
from PySide6.QtCore import Signal, Qt


class CalibrationButton(QWidget):
    overflow_requested = Signal()
    overflow_confirmed = Signal()
    ruler_requested    = Signal()

    _IDLE_STYLE = (
        "QPushButton { background-color: #3a3a5c; color: #c8d0e8; "
        "border: 1px solid #555580; padding: 4px 8px; }"
        "QPushButton:hover { background-color: #484870; }"
    )
    _ACTIVE_STYLE = (
        "QPushButton { background-color: #8B0000; color: white; "
        "border: 1px solid #cc2222; padding: 4px 8px; font-weight: bold; }"
        "QPushButton:hover { background-color: #a02020; }"
    )
    _ARROW_STYLE = (
        "QPushButton { background-color: #2e2e4a; color: #c8d0e8; "
        "border: 1px solid #555580; border-left: none; padding: 4px 3px; }"
        "QPushButton:hover { background-color: #484870; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._main_btn = QPushButton("Calibration")
        self._main_btn.setStyleSheet(self._IDLE_STYLE)

        self._arrow_btn = QPushButton("▾")
        self._arrow_btn.setFixedWidth(22)
        self._arrow_btn.setStyleSheet(self._ARROW_STYLE)
        self._arrow_btn.setCursor(QCursor(Qt.PointingHandCursor))

        layout.addWidget(self._main_btn)
        layout.addWidget(self._arrow_btn)

        self._active = False

        self._main_btn.clicked.connect(self._on_main_clicked)
        self._arrow_btn.clicked.connect(self._show_menu)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._main_btn.setText("✓  Confirm Overflow")
            self._main_btn.setStyleSheet(self._ACTIVE_STYLE)
        else:
            self._main_btn.setText("Calibration")
            self._main_btn.setStyleSheet(self._IDLE_STYLE)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _on_main_clicked(self):
        if self._active:
            self.set_active(False)
            self.overflow_confirmed.emit()

    def _show_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #1e1e2e; color: #c8d0e8; "
            "border: 1px solid #555580; }"
            "QMenu::item:selected { background-color: #3a3a5c; }"
            "QMenu::item:disabled { color: #555566; }"
        )

        act_overflow = QAction("⛶  Overflow Direction", self)
        act_ruler    = QAction("📏  Ruler  (coming soon)", self)
        act_ruler.setEnabled(False)

        menu.addAction(act_overflow)
        menu.addSeparator()
        menu.addAction(act_ruler)

        act_overflow.triggered.connect(self._on_overflow_chosen)

        menu.exec(QCursor.pos())

    def _on_overflow_chosen(self):
        self.set_active(True)
        self.overflow_requested.emit()
