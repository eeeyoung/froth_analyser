"""
macro_feature_widget.py — MacroFeatureWidget

Standalone pop-out window displaying 4 macroscopic image features as a
dynamically updating bar chart. Computed once per full frame (not per-ROI).

Features: Brightness (overall) + per-channel R, G, B mean pixel levels.

Visual design
-------------
- 4 color-coded bars with live value labels atop each bar.
- Delta arrows (↑/↓) show frame-to-frame trend per feature.
- Faint sparklines behind each bar show the last 30 frames of history.
- Color-matched axis labels replace the default monochrome X ticks.
"""
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont


_FEATURE_NAMES = ["Brightness", "R", "G", "B"]

# (fill_r, fill_g, fill_b) for bars — reused for sparklines and axis labels
_BAR_COLORS = [
    (200, 200, 200),   # Brightness — gray
    (255, 50, 50),     # R — red
    (50, 200, 50),     # G — green
    (50, 100, 255),    # B — blue
]

_SPARKLINE_LEN = 30   # frames of history per sparkline


class MacroFeatureWidget(QWidget):
    """Pop-out bar chart of full-frame macroscopic features.

    Always buffers incoming data regardless of visibility, so the chart
    is populated immediately when the user opens the window.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Macroscopic Image Features")
        self.resize(500, 420)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Per-feature rolling history for sparklines
        self._histories: list[deque[float]] = [
            deque(maxlen=_SPARKLINE_LEN) for _ in range(4)
        ]
        self._prev: list[float | None] = [None, None, None, None]

        # --- Plot ---
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0d1117")
        self._plot.setTitle("Macroscopic Image Features", color="#c9d1d9", size="11pt")
        self._plot.showGrid(y=True, alpha=0.15)
        self._plot.setLabel("left", "Pixel Value (0–255)", color="#8b949e", size="8pt")
        self._plot.setYRange(0, 295)  # headroom for labels above tall bars
        self._plot.getAxis("left").setWidth(48)
        self._plot.getAxis("bottom").setPen(pg.mkPen("#8b949e", width=1))

        # Hide default X-axis ticks — we draw coloured labels manually
        self._plot.getAxis("bottom").setTicks([[]])
        self._plot.getAxis("bottom").setStyle(showValues=False)
        self._plot.getAxis("bottom").setHeight(32)

        # --- Bars ---
        self._bars = pg.BarGraphItem(
            x=[0, 1, 2, 3],
            height=[0, 0, 0, 0],
            width=0.6,
            brushes=[pg.mkBrush(*c) for c in _BAR_COLORS],
            pen=pg.mkPen(width=0),
        )
        self._plot.addItem(self._bars)

        # --- Sparklines (behind bars, added first so they render under) ---
        self._sparklines: list[pg.PlotDataItem] = []
        for r, g, b in _BAR_COLORS:
            pen = pg.mkPen(r, g, b, 90, width=1.5)
            spark = pg.PlotDataItem(pen=pen, antialias=True)
            self._sparklines.append(spark)
            self._plot.addItem(spark)

        # --- Value labels atop each bar ---
        self._value_labels: list[pg.TextItem] = []
        for i in range(4):
            t = pg.TextItem("", anchor=(0.5, 0), color=(255, 255, 255))
            t.setFont(QFont("Arial", 9, QFont.Bold))
            self._value_labels.append(t)
            self._plot.addItem(t)

        # --- Delta (trend) labels above value labels ---
        self._delta_labels: list[pg.TextItem] = []
        for i in range(4):
            t = pg.TextItem("", anchor=(0.5, 0))
            t.setFont(QFont("Arial", 8))
            self._delta_labels.append(t)
            self._plot.addItem(t)

        # --- Colour-coded X-axis labels ---
        self._axis_labels: list[pg.TextItem] = []
        for i, (name, (r, g, b)) in enumerate(
            zip(_FEATURE_NAMES, _BAR_COLORS)
        ):
            t = pg.TextItem(name, anchor=(0.5, 1), color=(r, g, b))
            t.setFont(QFont("Arial", 10, QFont.Bold))
            self._axis_labels.append(t)
            self._plot.addItem(t)
            t.setPos(i, -2)

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
        values = [brightness, r_level, g_level, b_level]

        # Always update histories (regardless of visibility)
        for i, v in enumerate(values):
            self._histories[i].append(v)

        if not self.isVisible():
            self._prev = values
            return

        # --- Bars ---
        self._bars.setOpts(height=values)

        # --- Sparklines ---
        for i in range(4):
            hist = self._histories[i]
            if len(hist) >= 2:
                n = len(hist)
                xs = np.linspace(i - 0.28, i + 0.28, n)
                self._sparklines[i].setData(xs, list(hist))

        # --- Value + delta labels ---
        for i, v in enumerate(values):
            # Value
            self._value_labels[i].setText(f"{v:.1f}")
            self._value_labels[i].setPos(i, v + 8)

            # Delta
            prev = self._prev[i]
            if prev is None:
                self._delta_labels[i].setText("")
            else:
                delta = v - prev
                if abs(delta) < 0.05:
                    self._delta_labels[i].setText("—")
                    self._delta_labels[i].setColor((150, 150, 150))
                elif delta > 0:
                    self._delta_labels[i].setText(f"↑{delta:+.1f}")
                    self._delta_labels[i].setColor((100, 255, 100))
                else:
                    self._delta_labels[i].setText(f"↓{delta:+.1f}")
                    self._delta_labels[i].setColor((255, 100, 100))
            self._delta_labels[i].setPos(i, v + 26)

        self._prev = values

        # --- Simplified title ---
        avg = sum(values) / 4
        self._plot.setTitle(
            f"Macroscopic Image Features  —  Frame avg: {avg:.1f}",
            color="#c9d1d9",
            size="11pt",
        )

    def clear_data(self) -> None:
        self._histories = [deque(maxlen=_SPARKLINE_LEN) for _ in range(4)]
        self._prev = [None, None, None, None]
        self._bars.setOpts(height=[0, 0, 0, 0])
        for spark in self._sparklines:
            spark.setData([], [])
        for label in self._value_labels:
            label.setText("")
        for label in self._delta_labels:
            label.setText("")
        self._plot.setTitle("Macroscopic Image Features", color="#c9d1d9", size="11pt")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        self.hide()
        event.ignore()
