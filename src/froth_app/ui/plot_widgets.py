"""
plot_widgets.py
Reusable PyQtGraph plot widgets for analysis monitoring.
"""

from collections import deque
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QSizePolicy, QGraphicsEllipseItem
from PySide6.QtCore import Slot, Qt, QRectF

class LBPPlotWidget(pg.PlotWidget):
    """Live scrolling PCA scatter chart for one ROI."""

    HISTORY = 200

    def __init__(self, roi_index: int, parent=None):
        super().__init__(parent)

        self.roi_index = roi_index
        self.setBackground("#0d1117")
        self.setTitle(f"ROI {roi_index + 1}  —  LBP PCA",
                      color="#c9d1d9", size="8pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("left", "PC 2", color="#8b949e", size="8pt")
        self.setLabel("bottom", "PC 1", color="#8b949e", size="8pt")
        self.getAxis("bottom").setStyle(showValues=True)
        self.getAxis("left").setWidth(38)
        self.setMinimumHeight(350)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pc1 = deque(maxlen=self.HISTORY)
        self._pc2 = deque(maxlen=self.HISTORY)
        self._brushes = deque(maxlen=self.HISTORY)
        self._sizes = deque(maxlen=self.HISTORY)

        self._confidence_ellipse = QGraphicsEllipseItem()
        self._confidence_ellipse.setPen(pg.mkPen(color=(255, 255, 255, 80), width=1, style=Qt.DashLine))
        self._confidence_ellipse.hide()
        self.addItem(self._confidence_ellipse, ignoreBounds=True)
        
        self._fit_line = pg.PlotDataItem(pen=pg.mkPen(color=(100, 200, 255, 120), width=1, style=Qt.DashLine))
        self.addItem(self._fit_line, ignoreBounds=True)

        self._scatter = pg.ScatterPlotItem(pxMode=True, pen=pg.mkPen(None))
        self.addItem(self._scatter)

    @Slot(object)
    def push(self, data: dict):
        if data.get("is_baseline", True):
            elapsed = data.get("elapsed", 0.0)
            self.setTitle(f"ROI {self.roi_index + 1}  —  Baseline: {elapsed:.1f}s", color="#f1fa8c")
            return
            
        pc1 = data.get("pc1", 0.0)
        pc2 = data.get("pc2", 0.0)
        t_sq = data.get("t_squared", 0.0)
        is_anomaly = data.get("is_anomaly", False)

        self._pc1.append(pc1)
        self._pc2.append(pc2)
        
        if "mean_pc1" in data and not self._confidence_ellipse.isVisible():
            m1 = data["mean_pc1"]
            m2 = data["mean_pc2"]
            s1 = data["std_pc1"]
            s2 = data["std_pc2"]
            
            # 95% confidence chi-square threshold for 2 degrees of freedom is aprox 5.991
            threshold = 5.991
            a = s1 * np.sqrt(threshold)
            b = s2 * np.sqrt(threshold)
            
            self._confidence_ellipse.setRect(QRectF(m1 - a, m2 - b, 2 * a, 2 * b))
            self._confidence_ellipse.show()
            
            # Plot the principal eigenvector axis directly along PC1
            self._fit_line.setData([m1 - a, m1 + a], [m2, m2])
            
            # Dynamically attach variance labels to the physical axes
            if "var_pc1" in data and "var_pc2" in data:
                v1 = data["var_pc1"] * 100
                v2 = data["var_pc2"] * 100
                self.setLabel("bottom", f"PC 1 ({v1:.1f}%)", color="#8b949e", size="8pt")
                self.setLabel("left", f"PC 2 ({v2:.1f}%)", color="#8b949e", size="8pt")
            
        if is_anomaly:
            self._brushes.append(pg.mkBrush(255, 50, 50, 200)) # Larger deep red
            self._sizes.append(12)
        else:
            self._brushes.append(pg.mkBrush(100, 200, 255, 120)) # Small faint blue
            self._sizes.append(5)

        self._scatter.setData(
            x=list(self._pc1),
            y=list(self._pc2),
            brush=list(self._brushes),
            size=list(self._sizes)
        )
        
        status_color = "#ff5555" if is_anomaly else "#50fa7b"
        status_text = "ANOMALY!" if is_anomaly else "Normal"
        self.setTitle(f"ROI {self.roi_index + 1}  —  T²: {t_sq:.1f} ({status_text})", color=status_color)

    def clear_data(self):
        self._pc1.clear()
        self._pc2.clear()
        self._brushes.clear()
        self._sizes.clear()
        
        self._confidence_ellipse.hide()
        self._fit_line.setData([], [])
        
        self.setLabel("bottom", "PC 1", color="#8b949e", size="8pt")
        self.setLabel("left", "PC 2", color="#8b949e", size="8pt")
        
        self._scatter.setData(x=[], y=[], brush=[], size=[])
            
        self.setTitle(f"ROI {self.roi_index + 1}  —  LBP PCA", color="#c9d1d9")

    def set_roi_index(self, idx: int):
        self.roi_index = idx
        self.clear_data()

class VelocityPlotWidget(pg.PlotWidget):
    """Live scrolling Velocity chart for one ROI."""

    HISTORY = 60 # Store 60 seconds of history at 1 tick/second

    def __init__(self, roi_index: int, parent=None):
        super().__init__(parent)

        self.roi_index = roi_index
        self.setBackground("#0d1117")
        self.setTitle(f"ROI {roi_index + 1}  —  Velocity",
                      color="#c9d1d9", size="8pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("left", "Velocity", color="#8b949e", size="8pt")
        self.getAxis("bottom").setStyle(showValues=False)
        self.getAxis("left").setWidth(38)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._x = list(range(self.HISTORY))
        _zero = [0.0] * self.HISTORY
        self._velocity = deque(_zero, maxlen=self.HISTORY)

        self._curve = self.plot(
            self._x, list(self._velocity),
            pen=pg.mkPen("#f1fa8c", width=2.0),
            name="Velocity",
            fillLevel=0, fillBrush=(241, 250, 140, 50)
        )

    @Slot(object)
    def push(self, data: dict):
        vel = data.get("velocity", 0.0)
        unit = data.get("unit", "px")
        
        self._velocity.append(vel)
        self._curve.setData(self._x, list(self._velocity))
        
        self.setTitle(f"ROI {self.roi_index + 1}  —  Velocity: {vel:.4f} {unit}/s", color="#f1fa8c")

    def clear_data(self):
        zero = [0.0] * self.HISTORY
        self._velocity = deque(zero, maxlen=self.HISTORY)
        self._curve.setData(self._x, zero)
        self.setTitle(f"ROI {self.roi_index + 1}  —  Velocity", color="#c9d1d9")
