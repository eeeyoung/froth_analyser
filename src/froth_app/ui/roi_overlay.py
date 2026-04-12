import sys
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QApplication, QFileDialog, QGridLayout
)

# Ensure absolute imports work when running this file directly as a test script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from froth_app.core.video_source import VideoSource, VideoPlayerWidget
from froth_app.core.roi_manager import ROICoordinateManager

class ROIOverlayWidget(QWidget):
    """
    Transparent widget placed exactly on top of the VideoPlayerWidget.
    Intercepts mouse events to draw ROIs and maps coordinate scaling.
    """
    roi_finalized = Signal() # Emitted when a valid ROI is pushed into the manager

    def __init__(self, player_widget: VideoPlayerWidget, roi_manager: ROICoordinateManager, parent=None):
        super().__init__(parent)
        self.player_widget = player_widget
        self.roi_manager = roi_manager
        
        # Transparent background and let clicks pass through until we "enable drawing"
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent;")
        
        self.is_drawing = False
        self.drawing_enabled = False
        self.start_point = QPoint()
        self.current_rect = QRect()
        
        # Track active video dimensions to correctly map scaled label pixels -> OpenCV raw pixels
        self.last_frame_w = 1920
        self.last_frame_h = 1080
        
    def enable_drawing(self, enabled=True):
        """Toggle drawing logic. When True, catches the user's mouse clicks."""
        self.drawing_enabled = enabled
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not enabled)
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def update_frame_size(self, w, h):
        self.last_frame_w = w
        self.last_frame_h = h

    def mousePressEvent(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            if len(self.roi_manager.rois) >= self.roi_manager.max_rois:
                QMessageBox.warning(self, "Limit Reached", f"You can only draw {self.roi_manager.max_rois} ROIs.")
                self.enable_drawing(False)
                return

            self.is_drawing = True
            self.start_point = event.position().toPoint()
            self.current_rect = QRect(self.start_point, self.start_point)
            self.update() # Triggers paintEvent to show the box immediately

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            # Normalize corrects backwards dragging (e.g. dragging bottom-right to top-left)
            self.current_rect = QRect(self.start_point, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_enabled and self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.enable_drawing(False) # Turn off draw mode automatically after 1 rect

            # Delegate the mathematical reverse-scaling calculation
            img_rect = self.map_screen_to_image(self.current_rect)
            if img_rect:
                x, y, w, h = img_rect
                # Delegate business logic validation to the Core Engine
                success, msg = self.roi_manager.new_roi_coordinate(
                    x, y, w, h, self.last_frame_w, self.last_frame_h
                )
                if success:
                    self.roi_finalized.emit()
                else:
                    QMessageBox.warning(self, "Invalid ROI", msg)
            
            self.current_rect = QRect()
            self.update()

    def map_screen_to_image(self, rect):
        """Math to translate screen UI pixels dynamically back into OpenCV (X,Y) logic."""
        pixmap = self.player_widget.pixmap()
        if not pixmap or pixmap.isNull():
            return None
            
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        
        # Our PlayerWidget centers the video. Let's find the exact pixel offsets where video starts.
        x_offset = (label_w - pix_w) / 2
        y_offset = (label_h - pix_h) / 2
        
        scale_x = self.last_frame_w / pix_w
        scale_y = self.last_frame_h / pix_h
        
        img_x = int((rect.x() - x_offset) * scale_x)
        img_y = int((rect.y() - y_offset) * scale_y)
        img_w = int(rect.width() * scale_x)
        img_h = int(rect.height() * scale_y)
        
        return (img_x, img_y, img_w, img_h)

    def map_image_to_screen(self, img_x, img_y, img_w, img_h):
        """Translates exact OpenCV backend ROIs forward onto the UI wrapper."""
        pixmap = self.player_widget.pixmap()
        if not pixmap or pixmap.isNull():
            return None
            
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        
        x_offset = (label_w - pix_w) / 2
        y_offset = (label_h - pix_h) / 2
        
        scale_x = pix_w / self.last_frame_w
        scale_y = pix_h / self.last_frame_h
        
        scr_x = int(img_x * scale_x + x_offset)
        scr_y = int(img_y * scale_y + y_offset)
        scr_w = int(img_w * scale_x)
        scr_h = int(img_h * scale_y)
        
        return QRect(scr_x, scr_y, scr_w, scr_h)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw finalized ROIs (Confirmed by Core Engine)
        roi_pen = QPen(QColor(0, 255, 0)) # Neon green
        roi_pen.setWidth(2)
        painter.setPen(roi_pen)
        painter.setBrush(QBrush(QColor(0, 255, 0, 40))) # Green translucent overlay

        for i, (x, y, w, h) in enumerate(self.roi_manager.rois):
            scr_rect = self.map_image_to_screen(x, y, w, h)
            if scr_rect:
                painter.drawRect(scr_rect)
                painter.drawText(scr_rect.topLeft() + QPoint(4, 15), f"ROI {i+1}")

        # 2. Draw actively dragging Temporary ROI outline
        if self.is_drawing and not self.current_rect.isNull():
            draw_pen = QPen(QColor(255, 0, 0)) # Red dashed line while dragging
            draw_pen.setWidth(2)
            draw_pen.setStyle(Qt.DashLine)
            painter.setPen(draw_pen)
            painter.setBrush(QBrush(QColor(255, 0, 0, 40)))
            painter.drawRect(self.current_rect)


class CroppedROIWidget(QLabel):
    """Auxiliary UI module that receives an OpenCV array and displays just a small crop."""
    def __init__(self, title=""):
        super().__init__()
        self.setFixedSize(220, 220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #555; color: white;")
        self.setText(title)
        
    def update_crop(self, frame: np.ndarray, roi_rect_tuple):
        """Isolates the sub-pixel array for the Analysis Engine."""
        if frame is None or not roi_rect_tuple:
            self.clear()
            return
            
        x, y, w, h = roi_rect_tuple
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            return
            
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw, channels = rgb_crop.shape
        bytes_per_line = channels * cw
        
        q_img = QImage(rgb_crop.data, cw, ch, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ==========================================
# EXHAUSTIVE INTEGRATION TEST BLOCK
# ==========================================
if __name__ == "__main__":
    class IntegrationTestWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("RT-FFAT: ROI Overlay & Cropping Integration")
            self.resize(1150, 650) 

            # Initialize Core Engines
            self.video_source = VideoSource()
            self.roi_manager = ROICoordinateManager(max_rois=3)
            self.last_frame = None

            main_layout = QHBoxLayout(self)
            left_panel = QVBoxLayout()
            right_panel = QVBoxLayout()

            # --- Video + Transparency Layer Stacking ---
            self.canvas_container = QWidget()
            canvas_layout = QGridLayout(self.canvas_container)
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            
            self.player_widget = VideoPlayerWidget()
            self.overlay_widget = ROIOverlayWidget(self.player_widget, self.roi_manager)
            
            # Stacking trick: placing both widgets identically at (row 0, col 0)
            canvas_layout.addWidget(self.player_widget, 0, 0)
            canvas_layout.addWidget(self.overlay_widget, 0, 0)
            left_panel.addWidget(self.canvas_container, stretch=1)

            # Standard Playback Configs
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

            # Core Manager Buttons
            roi_btn_layout = QHBoxLayout()
            self.btn_add_roi = QPushButton("+ Add Custom ROI (Draw on Video)")
            self.btn_add_roi.setStyleSheet("background-color: #2F4F4F; color: white;")
            self.btn_undo_roi = QPushButton("- Undo ROI")
            self.btn_undo_roi.setStyleSheet("background-color: #8B0000; color: white;")
            roi_btn_layout.addWidget(self.btn_add_roi)
            roi_btn_layout.addWidget(self.btn_undo_roi)
            left_panel.addLayout(roi_btn_layout)

            # --- Live Processed Extractions ---
            self.crop_widgets = []
            right_panel.addWidget(QLabel("<h2>Live ROI Crops:</h2>"))
            for i in range(self.roi_manager.max_rois):
                cw = CroppedROIWidget(f"Empty Data\n(No ROI {i+1} found)")
                self.crop_widgets.append(cw)
                right_panel.addWidget(cw)
            right_panel.addStretch(1)

            main_layout.addLayout(left_panel, stretch=4)
            main_layout.addLayout(right_panel, stretch=1)

            # --- Signals Integration ---
            self.video_source.frame_ready.connect(self.process_new_frame)
            self.video_source.error_occurred.connect(self.show_error)
            
            self.btn_load_cam.clicked.connect(self.load_camera)
            self.btn_load_video.clicked.connect(self.load_video)
            self.btn_play.clicked.connect(self.video_source.play)
            self.btn_pause.clicked.connect(self.video_source.pause)
            
            # UI triggering ROI Interactions
            self.btn_add_roi.clicked.connect(lambda: self.overlay_widget.enable_drawing(True))
            self.btn_undo_roi.clicked.connect(self.undo_roi)
            self.overlay_widget.roi_finalized.connect(self.refresh_rois)

        def process_new_frame(self, frame: np.ndarray):
            """Master consumer slot driving the dashboard updates."""
            self.last_frame = frame
            
            h, w = frame.shape[:2]
            self.overlay_widget.update_frame_size(w, h)
            
            self.player_widget.receive_frame(frame)
            self.overlay_widget.update() # Keep drawing boxes
            
            # Offload heavy crops to secondary widgets
            for i in range(self.roi_manager.max_rois):
                if i < len(self.roi_manager.rois):
                    self.crop_widgets[i].update_crop(frame, self.roi_manager.rois[i])

        def undo_roi(self):
            self.roi_manager.remove_last_roi()
            self.refresh_rois()

        def refresh_rois(self):
            """Ensures side-panels refresh immediately even if playback is paused."""
            self.overlay_widget.update()
            # Wipe unused text panels
            for i in range(self.roi_manager.max_rois):
                if i >= len(self.roi_manager.rois):
                    self.crop_widgets[i].clear()
                    self.crop_widgets[i].setText(f"Empty Data\n(No ROI {i+1} found)")
            
            # Force trigger frame logic if user drew it while video is paused!
            if self.last_frame is not None and self.video_source._is_paused:
                self.process_new_frame(self.last_frame)

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
            self.video_source.release()
            super().closeEvent(event)

    app = QApplication(sys.argv)
    window = IntegrationTestWindow()
    window.show()
    sys.exit(app.exec())
