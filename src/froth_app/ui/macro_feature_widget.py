"""
macro_feature_widget.py — MacroFeatureWidget

Standalone pop-out window displaying 4 macroscopic image features as a
dynamically updating bar chart. Computed once per full frame (not per-ROI).

Features: Brightness (overall) + per-channel R, G, B mean pixel levels.
"""
from collections import deque

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent


class MacroFeatureWidget(QWidget):
    """Pop-out bar chart of full-frame macroscopic features.

    Always buffers incoming data regardless of visibility, so the chart
    is populated immediately when the user opens the window.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Macroscopic Image Features")
        self.resize(500, 400)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._history: deque[dict] = deque(maxlen=100)

        # --- Plot ---
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0d1117")
        self._plot.setTitle("Macroscopic Image Features", color="#c9d1d9", size="11pt")
        self._plot.showGrid(y=True, alpha=0.15)
        self._plot.setLabel("left", "Pixel Value (0–255)", color="#8b949e", size="8pt")
        self._plot.setYRange(0, 260)
        self._plot.getAxis("left").setWidth(48)

        ticks = [(0, "Brightness"), (1, "R"), (2, "G"), (3, "B")]
        self._plot.getAxis("bottom").setTicks([ticks])
        self._plot.getAxis("bottom").setStyle(showValues=True)
        self._plot.getAxis("bottom").setWidth(28)

        self._bars = pg.BarGraphItem(
            x=[0, 1, 2, 3],
            height=[0, 0, 0, 0],
            width=0.6,
            brushes=[
                pg.mkBrush(200, 200, 200),   # Brightness — gray
                pg.mkBrush(255, 50, 50),     # R — red
                pg.mkBrush(50, 200, 50),     # G — green
                pg.mkBrush(50, 100, 255),    # B — blue
            ],
            pen=pg.mkPen(width=0),
        )
        self._plot.addItem(self._bars)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_features(
        self,
        brightness: float,
        r_level: float,
        g_level: float,
        b_level: float,
    ) -> None:
        """Push one frame's features. Always buffers; paints only when visible."""
        self._history.append({
            "brightness": brightness,
            "r": r_level,
            "g": g_level,
            "b": b_level,
        })
        if not self.isVisible():
            return
        self._bars.setOpts(height=[brightness, r_level, g_level, b_level])
        self._plot.setTitle(
            f"Macroscopic Image Features  —  "
            f"Brg: {brightness:.1f}  |  "
            f"R: {r_level:.1f}  G: {g_level:.1f}  B: {b_level:.1f}",
            color="#c9d1d9",
            size="11pt",
        )

    def clear_data(self) -> None:
        self._history.clear()
        self._bars.setOpts(height=[0, 0, 0, 0])
        self._plot.setTitle("Macroscopic Image Features", color="#c9d1d9", size="11pt")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        self.hide()
        event.ignore()
