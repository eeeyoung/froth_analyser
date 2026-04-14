import sys
import os
import numpy as np
from collections import deque
from multiprocessing import freeze_support
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QApplication, QFileDialog, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Slot
import pyqtgraph as pg

# Allow imports if run directly from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.froth_app.core.video_source import VideoSource, VideoPlayerWidget
from src.froth_app.core.roi_manager import ROICoordinateManager
from src.froth_app.ui.roi_overlay import ROIOverlayWidget, CroppedROIWidget
from src.froth_app.core.calibration import CalibrationManager
from src.froth_app.core.data_hub import GlobalDataHub
from src.froth_app.engine.analyzer import AnalysisEngineMaster


# ---------------------------------------------------------------------------
# LBP Time-Series Plot Widget
# ---------------------------------------------------------------------------
class LBPPlotWidget(pg.PlotWidget):
    """
    Live scrolling time-series chart for one ROI's LBP-RGB metrics.

    Displays 4 lines over the last HISTORY frames:
        • Texture change score  — thick white  (dominant)
        • Peak LBP code — R     — red
        • Peak LBP code — G     — green
        • Peak LBP code — B     — blue

    Peak codes (0–255) are normalised to 0–1 so all series share the same
    Y axis. Change score is clamped to [0, 1] for display.
    """

    HISTORY = 120   # number of frames kept in the rolling buffer

    def __init__(self, roi_index: int, parent=None):
        super().__init__(parent)

        # --- Visual style ---
        self.setBackground("#0d1117")
        self.setTitle(f"ROI {roi_index + 1}  —  LBP Texture Metrics",
                      color="#c9d1d9", size="8pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setYRange(0, 1.05, padding=0)
        self.setLabel("left", "Value (norm.)", color="#8b949e", size="8pt")
        self.getAxis("bottom").setStyle(showValues=False)
        self.getAxis("left").setWidth(38)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Rolling data buffers ---
        self._x = list(range(self.HISTORY))
        _zero = [0.0] * self.HISTORY
        self._score  = deque(_zero, maxlen=self.HISTORY)
        self._peak_r = deque(_zero, maxlen=self.HISTORY)
        self._peak_g = deque(_zero, maxlen=self.HISTORY)
        self._peak_b = deque(_zero, maxlen=self.HISTORY)

        # --- Legend (must be added before plot() calls) ---
        legend = self.addLegend(
            offset=(-5, 5),
            labelTextColor="#c9d1d9",
            colCount=2,
        )
        legend.setParentItem(self.graphicsItem())

        # --- Curves --- (score plotted last so it renders on top)
        self._curve_r = self.plot(
            self._x, list(self._peak_r),
            pen=pg.mkPen("#ff6b6b", width=1.2),
            name="Peak R",
        )
        self._curve_g = self.plot(
            self._x, list(self._peak_g),
            pen=pg.mkPen("#51cf66", width=1.2),
            name="Peak G",
        )
        self._curve_b = self.plot(
            self._x, list(self._peak_b),
            pen=pg.mkPen("#74c0fc", width=1.2),
            name="Peak B",
        )
        self._curve_score = self.plot(
            self._x, list(self._score),
            pen=pg.mkPen("#ffffff", width=2.5),
            name="Texture Δ",
        )

    @Slot(float, int, int, int)
    def push(self, score: float, peak_r: int, peak_g: int, peak_b: int):
        """Append one frame's worth of data and refresh all curves."""
        self._score.append(min(float(score), 1.0))
        self._peak_r.append(peak_r / 255.0)
        self._peak_g.append(peak_g / 255.0)
        self._peak_b.append(peak_b / 255.0)

        self._curve_score.setData(self._x, list(self._score))
        self._curve_r.setData(self._x, list(self._peak_r))
        self._curve_g.setData(self._x, list(self._peak_g))
        self._curve_b.setData(self._x, list(self._peak_b))

    def clear_data(self):
        """Reset all buffers to zero (called when an ROI is removed)."""
        zero = [0.0] * self.HISTORY
        self._score  = deque(zero, maxlen=self.HISTORY)
        self._peak_r = deque(zero, maxlen=self.HISTORY)
        self._peak_g = deque(zero, maxlen=self.HISTORY)
        self._peak_b = deque(zero, maxlen=self.HISTORY)
        self._curve_score.setData(self._x, zero)
        self._curve_r.setData(self._x, zero)
        self._curve_g.setData(self._x, zero)
        self._curve_b.setData(self._x, zero)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class FullStackTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT-FFAT: Full Stack Multi-Processing Test")
        self.resize(1600, 700)

        # ==========================================
        # 1. Initialize All Core Engines
        # ==========================================
        self.calibration = CalibrationManager()
        self.data_hub = GlobalDataHub(self.calibration)
        self.analyzer = AnalysisEngineMaster(self.data_hub)
        
        self.video_source = VideoSource()
        self.roi_manager = ROICoordinateManager(max_rois=3)
        self.last_frame = None

        # Start the background data collector listener
        self.data_hub.start()

        # ==========================================
        # 2. Build UI Layout 
        # ==========================================
        main_layout = QHBoxLayout(self)
        left_panel  = QVBoxLayout()

        # --- Video Canvas Stack ---
        self.canvas_container = QWidget()
        canvas_layout = QGridLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.player_widget  = VideoPlayerWidget()
        self.overlay_widget = ROIOverlayWidget(self.player_widget, self.roi_manager)
        
        canvas_layout.addWidget(self.player_widget,  0, 0)
        canvas_layout.addWidget(self.overlay_widget, 0, 0)
        left_panel.addWidget(self.canvas_container, stretch=1)

        # --- Base Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_load_cam   = QPushButton("1. Camera")
        self.btn_load_video = QPushButton("2. Video")
        self.btn_play       = QPushButton("3. Play")
        self.btn_pause      = QPushButton("4. Pause")
        btn_layout.addWidget(self.btn_load_cam)
        btn_layout.addWidget(self.btn_load_video)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_pause)
        left_panel.addLayout(btn_layout)

        # --- ROI Architecture Buttons ---
        roi_btn_layout = QHBoxLayout()
        self.btn_add_roi = QPushButton("+ Add Custom ROI & Start Analysis Thread")
        self.btn_add_roi.setStyleSheet("background-color: #2F4F4F; color: white;")
        self.btn_undo_roi = QPushButton("- Undo ROI & Kill Thread")
        self.btn_undo_roi.setStyleSheet("background-color: #8B0000; color: white;")
        roi_btn_layout.addWidget(self.btn_add_roi)
        roi_btn_layout.addWidget(self.btn_undo_roi)
        left_panel.addLayout(roi_btn_layout)

        # ------------------------------------------------
        # Right Panel: [Crop thumbnail | LBP plot] per ROI
        # ------------------------------------------------
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel(
            "<h2>Live ROI Data</h2>"
            "<p style='color:gray;font-size:10px;'>"
            "Crop · LBP Texture Metrics (normalised)"
            "</p>"
        ))

        self.crop_widgets     = []
        self.lbp_plot_widgets = []

        for i in range(self.roi_manager.max_rois):
            # Crop thumbnail (existing widget)
            crop = CroppedROIWidget(f"Empty Data\n(No ROI {i+1} found)")
            crop.setFixedWidth(140)
            self.crop_widgets.append(crop)

            # LBP plot (new widget)
            plot = LBPPlotWidget(roi_index=i)
            self.lbp_plot_widgets.append(plot)

            # One row per ROI
            row = QHBoxLayout()
            row.addWidget(crop,  stretch=0)   # fixed-width thumbnail
            row.addWidget(plot,  stretch=1)   # expands to fill space
            right_panel.addLayout(row, stretch=1)

        right_panel.addStretch(0)

        main_layout.addLayout(left_panel,  stretch=3)
        main_layout.addLayout(right_panel, stretch=3)

        # ==========================================
        # 3. Signals & Hooks 
        # ==========================================
        self.video_source.frame_ready.connect(self.process_new_frame)
        self.video_source.error_occurred.connect(self.show_error)
        
        self.btn_load_cam.clicked.connect(self.load_camera)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.video_source.play)
        self.btn_pause.clicked.connect(self.video_source.pause)
        
        # User draws UI -> Maps math -> Triggers Analyzer Process Bootup
        self.btn_add_roi.clicked.connect(lambda: self.overlay_widget.enable_drawing(True))
        self.btn_undo_roi.clicked.connect(self.undo_roi)
        self.overlay_widget.roi_finalized.connect(self.on_roi_finalized)

        # Wire DataHub LBP signal -> plot update (Qt queues this safely across threads)
        self.data_hub.lbp_data_ready.connect(self.on_lbp_data)

    # ==========================================
    # 4. Multiprocessing Event Managers
    # ==========================================
    def on_roi_finalized(self):
        """Called when user finishes drawing a valid green box."""
        new_roi_id = len(self.roi_manager.rois) - 1 
        
        # Tell the engine to spin up a background CPU core for this ID
        self.analyzer.add_roi_stream(new_roi_id, [2])
        
        # Update UI graphics
        self.refresh_rois()

    def undo_roi(self):
        """Called when user deletes a box."""
        if len(self.roi_manager.rois) > 0:
            last_roi_id = len(self.roi_manager.rois) - 1
            
            # Send Poison Pill to murder the background process
            self.analyzer.remove_roi_stream(last_roi_id)
            
            # Clear that ROI's plot data
            self.lbp_plot_widgets[last_roi_id].clear_data()
            
            # Delete the coordinates from the logic manager
            self.roi_manager.remove_last_roi()
            
            # Update UI graphics
            self.refresh_rois()

    def process_new_frame(self, frame: np.ndarray):
        """Master frame dispatcher."""
        self.last_frame = frame
        
        # 1. Update Video Renderer
        h, w = frame.shape[:2]
        self.overlay_widget.update_frame_size(w, h)
        self.player_widget.receive_frame(frame)
        self.overlay_widget.update()
        
        # 2. Update Side UI Crops
        for i in range(self.roi_manager.max_rois):
            if i < len(self.roi_manager.rois):
                self.crop_widgets[i].update_crop(frame, self.roi_manager.rois[i])
                
        # 3. DISPATCH FRAME TO BACKGROUND ANALYSIS PROCESSES
        self.analyzer.process_frame(frame, self.roi_manager.rois)

    @Slot(int, object)
    def on_lbp_data(self, roi_id: int, result: dict):
        """Receive LBP result from DataHub and push it to the right plot."""
        if roi_id >= len(self.lbp_plot_widgets):
            return

        hist_r = result.get("lbp_r_hist")
        hist_g = result.get("lbp_g_hist")
        hist_b = result.get("lbp_b_hist")
        score  = result.get("texture_change_score", 0.0)

        peak_r = int(hist_r.argmax()) if hist_r is not None else 0
        peak_g = int(hist_g.argmax()) if hist_g is not None else 0
        peak_b = int(hist_b.argmax()) if hist_b is not None else 0

        self.lbp_plot_widgets[roi_id].push(score, peak_r, peak_g, peak_b)

    def refresh_rois(self):
        self.overlay_widget.update()
        for i in range(self.roi_manager.max_rois):
            if i >= len(self.roi_manager.rois):
                self.crop_widgets[i].clear()
                self.crop_widgets[i].setText(f"Empty Data\n(No ROI {i+1} found)")
        
        if self.last_frame is not None and self.video_source._is_paused:
            self.process_new_frame(self.last_frame)

    # ==========================================
    # Utilities
    # ==========================================
    def load_camera(self):
        cameras = VideoSource.get_camera_sources()
        if cameras:
            if self.video_source.load_source(cameras[0]):
                self.video_source.play()
        else:
            self.show_error("No cameras detected.")

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            if self.video_source.load_source(file_path):
                self.video_source.play()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """CRITICAL: Safely kill all multi-processes to prevent memory leaks."""
        self.analyzer.shutdown_all()
        self.data_hub.shutdown()
        self.video_source.release()
        super().closeEvent(event)


if __name__ == "__main__":
    # Required for Windows multiprocessing to not endlessly fork application
    freeze_support() 
    
    app = QApplication(sys.argv)
    window = FullStackTestWindow()
    window.show()
    sys.exit(app.exec())
