import sys
import os
import numpy as np
from multiprocessing import freeze_support
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QApplication, QFileDialog, QGridLayout
)

# Allow imports if run directly from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.froth_app.core.video_source import VideoSource, VideoPlayerWidget
from src.froth_app.core.roi_manager import ROICoordinateManager
from src.froth_app.ui.roi_overlay import ROIOverlayWidget, CroppedROIWidget
from src.froth_app.core.calibration import CalibrationManager
from src.froth_app.core.data_hub import GlobalDataHub
from src.froth_app.engine.analyzer import AnalysisEngineMaster

class FullStackTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT-FFAT: Full Stack Multi-Processing Test")
        self.resize(1150, 650) 

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
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # --- Video Canvas Stack ---
        self.canvas_container = QWidget()
        canvas_layout = QGridLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.player_widget = VideoPlayerWidget()
        self.overlay_widget = ROIOverlayWidget(self.player_widget, self.roi_manager)
        
        canvas_layout.addWidget(self.player_widget, 0, 0)
        canvas_layout.addWidget(self.overlay_widget, 0, 0)
        left_panel.addWidget(self.canvas_container, stretch=1)

        # --- Base Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_load_cam = QPushButton("1. Camera")
        self.btn_load_video = QPushButton("2. Video")
        self.btn_play = QPushButton("3. Play")
        self.btn_pause = QPushButton("4. Pause")
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

        # --- Right Panel (Crop Visualization) ---
        self.crop_widgets = []
        right_panel.addWidget(QLabel("<h2>Live Visual Crops:</h2><p>(Tracking data prints to terminal)</p>"))
        for i in range(self.roi_manager.max_rois):
            cw = CroppedROIWidget(f"Empty Data\n(No ROI {i+1} found)")
            self.crop_widgets.append(cw)
            right_panel.addWidget(cw)
        right_panel.addStretch(1)

        main_layout.addLayout(left_panel, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

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

    # ==========================================
    # 4. Multiprocessing Event Managers
    # ==========================================
    def on_roi_finalized(self):
        """Called when user finishes drawing a valid green box."""
        # The new ROI is always the last one appended to the list
        new_roi_id = len(self.roi_manager.rois) - 1 
        
        # Tell the engine to spin up a background CPU core for this ID
        self.analyzer.add_roi_stream(new_roi_id)
        
        # Update UI graphics
        self.refresh_rois()

    def undo_roi(self):
        """Called when user deletes a box."""
        if len(self.roi_manager.rois) > 0:
            last_roi_id = len(self.roi_manager.rois) - 1
            
            # Send Poison Pill to murder the background process
            self.analyzer.remove_roi_stream(last_roi_id)
            
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
