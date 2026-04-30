import sys
import os
import numpy as np
from collections import deque
from multiprocessing import freeze_support
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QMessageBox, QApplication, QFileDialog,
    QGridLayout, QSizePolicy, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QStackedWidget, QGraphicsEllipseItem
)
from PySide6.QtCore import Slot, Qt, QRectF
import pyqtgraph as pg

# Allow imports if run directly from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.froth_app.core.video_source import VideoSource, VideoPlayerWidget
from src.froth_app.core.roi_manager import ROICoordinateManager
from src.froth_app.ui.roi_overlay import ROIOverlayWidget, CroppedROIWidget
from src.froth_app.ui.roi_detail_window import ROIDetailWindow
from src.froth_app.ui.log_book_interface import LogBookInterface
from src.froth_app.ui.functions_dialog import FunctionsDialog
from src.froth_app.ui.plot_widgets import LBPPlotWidget, VelocityPlotWidget, TSquarePlotWidget, QStatisticPlotWidget
from src.froth_app.ui.overflow_calibration_widget import OverflowCalibrationWidget
from src.froth_app.ui.calibration_button import CalibrationButton
from src.froth_app.core.calibration import CalibrationManager
from src.froth_app.core.data_hub import GlobalDataHub
from src.froth_app.core.algorithm_state import AlgorithmStateManager
from src.froth_app.engine.analyzer import AnalysisEngineMaster

# Algorithm IDs (must match ALGORITHM_REGISTRY in analyzer.py)
_ALGO_LK  = 1
_ALGO_LBP = 2





# ---------------------------------------------------------------------------
# Live Data Config Dialog
# ---------------------------------------------------------------------------
class LiveDataConfigDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Data Configuration")
        
        layout = QVBoxLayout(self)
        
        self.chk_all = QCheckBox("All live data")
        self.chk_lbp = QCheckBox("LBP plot")
        self.chk_t2 = QCheckBox("T² plot")
        self.chk_q = QCheckBox("Q-statistic plot")
        self.chk_vel = QCheckBox("Velocity plot")
        
        self.chk_all.setChecked(current_config.get("all", True))
        self.chk_lbp.setChecked(current_config.get("lbp", True))
        self.chk_t2.setChecked(current_config.get("t2", True))
        self.chk_q.setChecked(current_config.get("q", True))
        self.chk_vel.setChecked(current_config.get("vel", True))
        
        layout.addWidget(self.chk_all)
        layout.addWidget(self.chk_lbp)
        layout.addWidget(self.chk_t2)
        layout.addWidget(self.chk_q)
        layout.addWidget(self.chk_vel)
        
        self.chk_all.toggled.connect(self._on_all_toggled)
        self._on_all_toggled(self.chk_all.isChecked())
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    def _on_all_toggled(self, checked):
        self.chk_lbp.setEnabled(not checked)
        self.chk_t2.setEnabled(not checked)
        self.chk_q.setEnabled(not checked)
        self.chk_vel.setEnabled(not checked)
        if checked:
            self.chk_lbp.setChecked(True)
            self.chk_t2.setChecked(True)
            self.chk_q.setChecked(True)
            self.chk_vel.setChecked(True)

    def get_config(self):
        return {
            "all": self.chk_all.isChecked(),
            "lbp": self.chk_lbp.isChecked(),
            "t2": self.chk_t2.isChecked(),
            "q": self.chk_q.isChecked(),
            "vel": self.chk_vel.isChecked()
        }

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class FullStackTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT-FFAT: Full Stack Multi-Processing Test")
        self.resize(1600, 900)

        # ==========================================
        # 1. Initialize All Core Engines
        # ==========================================
        self.calibration  = CalibrationManager()
        self.algo_state   = AlgorithmStateManager()
        self.data_hub     = GlobalDataHub(self.calibration)
        self.analyzer     = AnalysisEngineMaster(self.data_hub)
        
        self.log_book_widget = LogBookInterface()
        self.data_hub.log_book.log_ready.connect(self.log_book_widget.push_log)

        self.video_source = VideoSource()
        self.roi_manager  = ROICoordinateManager(max_rois=3)
        self.last_frame   = None
        self._is_playing  = False
        
        self._live_config = {
            "all": True,
            "lbp": True,
            "t2": True,
            "q": True,
            "vel": True
        }

        # Last raw crop per ROI — updated every frame and forwarded to detail windows
        self._last_crops: list[np.ndarray | None] = [None] * self.roi_manager.max_rois

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
        self.calib_widget   = OverflowCalibrationWidget(self.calibration)

        canvas_layout.addWidget(self.player_widget,  0, 0)
        canvas_layout.addWidget(self.overlay_widget, 0, 0)
        canvas_layout.addWidget(self.calib_widget,   0, 0)
        left_panel.addWidget(self.canvas_container, stretch=1)

        # --- Base Buttons ---
        btn_layout = QHBoxLayout()

        self.btn_load_cam   = QPushButton("1. Camera")
        self.btn_load_video = QPushButton("2. Video")
        self.btn_play_pause = QPushButton("3. Play")
        self.btn_calibration = CalibrationButton()

        self.btn_functions = QPushButton("4. Functions")
        self.btn_functions.setStyleSheet("background-color: #555555; color: white;")
        
        self.btn_logbook = QPushButton("5. Log Book")
        self.btn_logbook.setStyleSheet("background-color: #2F4F4F; color: white;")

        btn_layout.addWidget(self.btn_load_cam)
        btn_layout.addWidget(self.btn_load_video)
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_calibration)
        btn_layout.addWidget(self.btn_functions)
        btn_layout.addWidget(self.btn_logbook)
        left_panel.addLayout(btn_layout)

        # --- ROI Architecture Buttons ---
        roi_btn_layout = QHBoxLayout()
        self.btn_add_roi  = QPushButton("+ Add Custom ROI & Start Analysis Thread")
        self.btn_add_roi.setStyleSheet("background-color: #2F4F4F; color: white;")
        self.btn_undo_roi = QPushButton("- Undo ROI & Kill Thread")
        self.btn_undo_roi.setStyleSheet("background-color: #8B0000; color: white;")
        roi_btn_layout.addWidget(self.btn_add_roi)
        roi_btn_layout.addWidget(self.btn_undo_roi)
        left_panel.addLayout(roi_btn_layout)

        # ------------------------------------------------
        # Right Panel: header + "Show live data" checkbox
        #              then [thumbnail | LBP plot] per ROI
        # ------------------------------------------------
        right_panel = QVBoxLayout()

        right_panel.addWidget(QLabel(
            "<h2>Live ROI Data</h2>"
            "<p style='color:gray;font-size:10px;'>Click a thumbnail to open the live detail view</p>"
        ))

        self.btn_live_config = QPushButton("Live data Config")
        self.btn_live_config.setStyleSheet("background-color: #2F4F4F; color: white;")
        right_panel.addWidget(self.btn_live_config)

        self._selected_roi = 0

        self.crop_widgets     = []
        self.lbp_plot_widgets = []
        self.t2_plot_widgets  = []
        self.q_plot_widgets   = []
        self.lk_plot_widgets  = []
        self._detail_windows: list[ROIDetailWindow | None] = []
        
        thumbnails_layout = QHBoxLayout()
        self.plot_stack = QStackedWidget()
        self.t2_plot_stack = QStackedWidget()
        self.q_plot_stack = QStackedWidget()
        self.vel_plot_stack = QStackedWidget()

        for i in range(self.roi_manager.max_rois):
            # Thumbnail selector
            crop = CroppedROIWidget(f"Empty Data\n(No ROI {i+1} found)\nDouble-click to open live view")
            crop.setFixedWidth(140)
            self.crop_widgets.append(crop)
            thumbnails_layout.addWidget(crop)

            # Detail popup — created once, shown/hidden on click
            detail = ROIDetailWindow(
                roi_index=i,
                lk_active=self.algo_state.is_active(_ALGO_LK),
            )
            self._detail_windows.append(detail)
            
            # Independent LBP chart for THIS ROI placed directly into the invisible stack
            plot = LBPPlotWidget(roi_index=i)
            self.lbp_plot_widgets.append(plot)
            self.plot_stack.addWidget(plot)
            
            # T-squared chart
            t2plot = TSquarePlotWidget(roi_index=i)
            self.t2_plot_widgets.append(t2plot)
            self.t2_plot_stack.addWidget(t2plot)
            
            # Q-statistic chart
            qplot = QStatisticPlotWidget(roi_index=i)
            self.q_plot_widgets.append(qplot)
            self.q_plot_stack.addWidget(qplot)
            
            # Independent Velocity chart for THIS ROI placed directly into the invisible stack
            vplot = VelocityPlotWidget(roi_index=i)
            self.lk_plot_widgets.append(vplot)
            self.vel_plot_stack.addWidget(vplot)

            # Wire thumbnail clicks
            crop.clicked.connect(lambda checked=False, idx=i: self.select_roi(idx))
            crop.double_clicked.connect(lambda checked=False, idx=i: self._open_detail(idx))

        thumbnails_layout.addStretch()

        right_panel.addLayout(thumbnails_layout)
        right_panel.addWidget(self.plot_stack, stretch=3)
        right_panel.addWidget(self.t2_plot_stack, stretch=1)
        right_panel.addWidget(self.q_plot_stack, stretch=1)
        right_panel.addWidget(self.vel_plot_stack, stretch=1)

        self.select_roi(0)

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
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_functions.clicked.connect(self.open_functions_dialog)
        self.btn_logbook.clicked.connect(self.open_log_book)

        # Calibration button signals
        self.btn_calibration.overflow_requested.connect(self._start_overflow_calibration)
        self.btn_calibration.overflow_confirmed.connect(self._confirm_overflow_calibration)

        self.btn_add_roi.clicked.connect(self._try_add_roi)
        self.btn_undo_roi.clicked.connect(self.undo_roi)
        self.overlay_widget.roi_finalized.connect(self.on_roi_finalized)

        self.btn_live_config.clicked.connect(self.open_live_data_config)

        self.data_hub.lbp_data_ready.connect(self.on_lbp_data)
        self.data_hub.lk_data_ready.connect(self.on_lk_data)

    # ==========================================
    # 4. Button Slot Handlers
    # ==========================================
    @Slot()
    def toggle_play_pause(self):
        if self._is_playing:
            self.video_source.pause()
            self.btn_play_pause.setText("3. Play")
            self._is_playing = False
        else:
            self.video_source.play()
            self.btn_play_pause.setText("3. Pause")
            self._is_playing = True

    @Slot()
    def open_functions_dialog(self):
        """Open the Functions selection popup and apply changes on confirm."""
        dlg = FunctionsDialog(algo_state=self.algo_state, data_hub=self.data_hub, parent=self)
        dlg.exec()
        # Sync crosshair visibility in all open detail windows
        lk_active = self.algo_state.is_active(_ALGO_LK)
        for detail in self._detail_windows:
            if detail is not None:
                detail.set_lk_visible(lk_active)

    @Slot()
    def open_log_book(self):
        """Open the realtime visual log book dashboard."""
        self.log_book_widget.show()
        self.log_book_widget.raise_()
        self.log_book_widget.activateWindow()

    @Slot()
    def open_live_data_config(self):
        dlg = LiveDataConfigDialog(self._live_config, parent=self)
        if dlg.exec():
            self._live_config = dlg.get_config()
            self._apply_live_config()

    def _apply_live_config(self):
        self.plot_stack.setVisible(self._live_config["lbp"])
        self.t2_plot_stack.setVisible(self._live_config["t2"])
        self.q_plot_stack.setVisible(self._live_config["q"])
        self.vel_plot_stack.setVisible(self._live_config["vel"])
        
        any_visible = any([
            self._live_config["lbp"], 
            self._live_config["t2"], 
            self._live_config["q"], 
            self._live_config["vel"]
        ])
        for crop in self.crop_widgets:
            crop.setVisible(any_visible)

    def select_roi(self, idx: int):
        self._selected_roi = idx
        for i, crop in enumerate(self.crop_widgets):
            crop.set_selected(i == idx)
        self.plot_stack.setCurrentIndex(idx)
        self.t2_plot_stack.setCurrentIndex(idx)
        self.q_plot_stack.setCurrentIndex(idx)
        self.vel_plot_stack.setCurrentIndex(idx)

    def _open_detail(self, roi_id: int):
        """Show or raise the detail popup for the given ROI."""
        detail = self._detail_windows[roi_id]
        if detail is None:
            return
        # If there's already a cached crop, populate the window immediately
        if self._last_crops[roi_id] is not None:
            detail.update_frame(self._last_crops[roi_id])
        detail.show()
        detail.raise_()
        detail.activateWindow()

    # ==========================================
    # 5. Multiprocessing Event Managers
    # ==========================================
    # ==========================================
    # Calibration Handlers
    # ==========================================
    def _start_overflow_calibration(self):
        """Show the compass overlay for overflow direction calibration."""
        self.calib_widget.activate()

    def _confirm_overflow_calibration(self):
        """User confirmed — store calibration and hide the overlay."""
        self.calibration.confirm_overflow()
        self.calib_widget.deactivate()

    def _try_add_roi(self):
        """Guard ROI drawing: requires overflow calibration first."""
        if not self.calibration.overflow_calibrated:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Calibration Required",
                "Please calibrate the overflow direction before drawing ROIs.\n\n"
                "Click  Calibration → Overflow Direction,  set the needle, "
                "then click  ✓ Confirm Overflow."
            )
            return
        self.overlay_widget.enable_drawing(True)

    def on_roi_finalized(self):
        """Called when user finishes drawing a valid green box."""
        new_roi_id = len(self.roi_manager.rois) - 1
        active_ids = self.algo_state.active_ids()
        self.analyzer.add_roi_stream(new_roi_id, active_ids)
        self.refresh_rois()

    def undo_roi(self):
        if len(self.roi_manager.rois) > 0:
            last_roi_id = len(self.roi_manager.rois) - 1
            self.analyzer.remove_roi_stream(last_roi_id)
            self.lbp_plot_widgets[last_roi_id].clear_data()
            self.t2_plot_widgets[last_roi_id].clear_data()
            self.q_plot_widgets[last_roi_id].clear_data()
            self.lk_plot_widgets[last_roi_id].clear_data()
            self._last_crops[last_roi_id] = None
            detail = self._detail_windows[last_roi_id]
            if detail is not None:
                detail.reset()
                detail.hide()
            self.roi_manager.remove_last_roi()
            self.refresh_rois()

    def process_new_frame(self, frame: np.ndarray):
        self.last_frame = frame

        h, w = frame.shape[:2]
        self.overlay_widget.update_frame_size(w, h)
        self.player_widget.receive_frame(frame)
        self.overlay_widget.update()

        for i in range(self.roi_manager.max_rois):
            if i < len(self.roi_manager.rois):
                x, y, rw, rh = self.roi_manager.rois[i]
                crop = frame[y:y + rh, x:x + rw]

                # Update the small thumbnail
                self.crop_widgets[i].update_crop(frame, self.roi_manager.rois[i])

                # Cache and forward to detail window (only if it is visible)
                self._last_crops[i] = crop.copy()
                detail = self._detail_windows[i]
                if detail is not None and detail.isVisible():
                    detail.update_frame(crop)

        self.analyzer.process_frame(frame, self.roi_manager.rois)

    @Slot(int, object)
    def on_lbp_data(self, roi_id: int, processed: dict):
        if roi_id < len(self.lbp_plot_widgets):
            self.lbp_plot_widgets[roi_id].push(processed)
            self.t2_plot_widgets[roi_id].push(processed)
            self.q_plot_widgets[roi_id].push(processed)

    @Slot(int, object)
    def on_lk_data(self, roi_id: int, processed: dict):
        """Receive processed LK result and update the detail window crosshair."""
        if roi_id >= len(self._detail_windows):
            return
            
        if processed.get("velocity_ready", False):
            if roi_id < len(self.lk_plot_widgets):
                self.lk_plot_widgets[roi_id].push(processed)
                
        detail = self._detail_windows[roi_id]
        if detail is not None and detail.isVisible():
            dx = processed.get("dx_pixels", 0.0)
            dy = processed.get("dy_pixels", 0.0)
            detail.update_lk(dx, dy)

    def refresh_rois(self):
        self.overlay_widget.update()
        for i in range(self.roi_manager.max_rois):
            if i >= len(self.roi_manager.rois):
                self.crop_widgets[i].clear()
                self.crop_widgets[i].setText(
                    f"Empty Data\n(No ROI {i+1} found)\nDouble-click to open live view"
                )

        if self.last_frame is not None and self.video_source._is_paused:
            self.process_new_frame(self.last_frame)

    # ==========================================
    # Utilities
    # ==========================================
    def load_camera(self):
        cameras = VideoSource.get_camera_sources()
        if cameras:
            if self.video_source.load_source(cameras[0]):
                self.toggle_play_pause()
        else:
            self.show_error("No cameras detected.")

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            if self.video_source.load_source(file_path):
                self.toggle_play_pause()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """CRITICAL: Safely kill all multi-processes to prevent memory leaks."""
        # Force-close all detail windows so they don't linger
        for detail in self._detail_windows:
            if detail is not None:
                detail.setAttribute(Qt.WA_DeleteOnClose, True)
                detail.close()
        self.analyzer.shutdown_all()
        self.data_hub.log_book.flush_and_close()
        self.data_hub.shutdown()
        self.video_source.release()
        super().closeEvent(event)


if __name__ == "__main__":
    if sys.platform == "win32":
        import ctypes
        # Force 1ms timer resolution on Windows to prevent msleep() stutter
        ctypes.WinDLL('winmm').timeBeginPeriod(1)

    freeze_support()
    app = QApplication(sys.argv)
    window = FullStackTestWindow()
    window.show()
    sys.exit(app.exec())
