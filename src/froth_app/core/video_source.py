import sys
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, Qt, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)

# ---------------------------------------------------------------------------
# Optional decord import — graceful fallback to cv2 if not installed
# ---------------------------------------------------------------------------
_HAS_DECORD = False
_HAS_DECORD_GPU = False
try:
    from decord import VideoReader as DecordVideoReader
    from decord import cpu as decord_cpu, gpu as decord_gpu
    _HAS_DECORD = True

    # Probe GPU availability at import time with a safe check.
    # decord raises RuntimeError if CUDA is not available.
    try:
        # A lightweight probe — no actual video needed
        _ = decord_gpu(0)
        _HAS_DECORD_GPU = True
    except Exception:
        _HAS_DECORD_GPU = False

    _backend_label = "decord (GPU)" if _HAS_DECORD_GPU else "decord (CPU)"
    print(f"[VideoSource] Hardware decoding available: {_backend_label}")
except ImportError:
    print("[VideoSource] decord not installed — using OpenCV CPU decoding.")


class VideoSource(QThread):
    """
    Producer thread that handles video frames acquisition from a camera or a video file.
    Runs asynchronously and emits frames to the UI consumer.

    Video file decoding priority (cross-platform)
    -----------------------------------------------
    1. decord + GPU   — Windows with Nvidia GPU  (NVDEC hardware decoder)
    2. decord + CPU   — macOS / Windows without GPU  (still faster than cv2)
    3. cv2.VideoCapture — fallback if decord is not installed

    Live camera sources always use cv2.VideoCapture (decord does not support webcams).
    """
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = None          # cv2.VideoCapture (cameras or fallback)
        self._decord_vr = None       # decord.VideoReader (video files)
        self._decord_pos = 0         # Current frame index for decord
        self._decord_total = 0       # Total frame count from decord
        self._is_running = False
        self._is_paused = False
        self.fps = 30.0
        self.is_file = False
        self._using_decord = False   # True when decord is the active backend
        self.mutex = QMutex()

    @staticmethod
    def get_camera_sources(max_tests=3):
        """
        Tests and returns a list of available local webcam indices.
        Uses CAP_DSHOW on Windows for significantly faster initialization.
        """
        available_cameras = []
        for i in range(max_tests):
            if sys.platform == "win32":
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
                
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        print(available_cameras)
        return available_cameras

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _release_current(self):
        """Release whichever backend is currently active."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._decord_vr = None
        self._decord_pos = 0
        self._decord_total = 0
        self._using_decord = False

    def load_source(self, source):
        """
        Loads a camera index (int) or a video file path (str).

        For video files, attempts to use decord for hardware-accelerated
        decoding before falling back to cv2.VideoCapture.
        """
        with QMutexLocker(self.mutex):
            self._release_current()
            
            # Integers are usually local cameras; strings are files/streams
            if isinstance(source, int):
                # Live camera — always use cv2
                if sys.platform == "win32":
                    self.capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    self.capture = cv2.VideoCapture(source)
                self.is_file = False

                if not self.capture.isOpened():
                    self.error_occurred.emit(f"Failed to open camera: {source}")
                    return False

                # Camera FPS
                extracted_fps = self.capture.get(cv2.CAP_PROP_FPS)
                if extracted_fps > 0 and not np.isnan(extracted_fps):
                    self.fps = extracted_fps
                else:
                    self.fps = 30.0

            else:
                # Video file — try decord first, then fall back to cv2
                self.is_file = True

                if _HAS_DECORD:
                    try:
                        ctx = decord_gpu(0) if _HAS_DECORD_GPU else decord_cpu(0)
                        self._decord_vr = DecordVideoReader(source, ctx=ctx)
                        self._decord_total = len(self._decord_vr)
                        self._decord_pos = 0
                        self._using_decord = True

                        # Extract FPS from decord
                        avg_fps = self._decord_vr.get_avg_fps()
                        self.fps = avg_fps if avg_fps > 0 else 30.0

                        backend = "GPU (NVDEC)" if _HAS_DECORD_GPU else "CPU"
                        print(
                            f"[VideoSource] Opened '{source}' with decord {backend} | "
                            f"{self._decord_total} frames @ {self.fps:.1f} FPS"
                        )
                        return True
                    except Exception as e:
                        print(
                            f"[VideoSource] decord failed for '{source}': {e} — "
                            f"falling back to OpenCV."
                        )
                        self._decord_vr = None
                        self._using_decord = False

                # Fallback: cv2.VideoCapture
                self.capture = cv2.VideoCapture(source)
                if not self.capture.isOpened():
                    self.error_occurred.emit(f"Failed to open source: {source}")
                    return False

                extracted_fps = self.capture.get(cv2.CAP_PROP_FPS)
                if extracted_fps > 0 and not np.isnan(extracted_fps):
                    self.fps = extracted_fps
                else:
                    self.fps = 30.0

                print(f"[VideoSource] Opened '{source}' with OpenCV CPU @ {self.fps:.1f} FPS")

        return True

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """The main thread loop that reads and emits frames."""
        self._is_running = True
        
        while self._is_running:
            delay = 1  # Milliseconds to sleep per loop iteration (minimum)

            with QMutexLocker(self.mutex):
                if not self._is_paused:
                    if self._using_decord and self._decord_vr is not None:
                        frame = self._read_decord()
                        if frame is not None:
                            self.frame_ready.emit(frame)
                            delay = int(1000 / self.fps)
                    elif self.capture and self.capture.isOpened():
                        ret, frame = self.capture.read()
                        if ret:
                            self.frame_ready.emit(frame)
                            if self.is_file:
                                delay = int(1000 / self.fps)
                        else:
                            if self.is_file:
                                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            else:
                                self._is_paused = True

            # Sleep is called outside the mutex lock so the UI can still interact (pause, play)
            if self._is_paused:
                self.msleep(50)  # Sleep longer when paused to save CPU
            else:
                self.msleep(delay)

    def _read_decord(self) -> np.ndarray | None:
        """
        Read the next frame from the decord VideoReader.

        Returns a BGR numpy array (matching cv2 convention) or None at EOF.
        Automatically loops back to the start when the video ends.
        """
        if self._decord_pos >= self._decord_total:
            # Loop back to start
            self._decord_pos = 0

        try:
            # decord returns RGB; convert to BGR to match the rest of the pipeline
            rgb_frame = self._decord_vr[self._decord_pos].asnumpy()
            self._decord_pos += 1
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        except Exception:
            self._decord_pos = 0
            return None

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self):
        with QMutexLocker(self.mutex):
            self._is_paused = False
        if not self.isRunning():
            self.start()

    def pause(self):
        with QMutexLocker(self.mutex):
            self._is_paused = True

    def stop(self):
        with QMutexLocker(self.mutex):
            self._is_running = False
            self._is_paused = False

    def release(self):
        """Ensure thread stops safely and resources are released."""
        self.stop()
        self.wait()  # Block until the thread loop exits naturally
        self._release_current()


class VideoPlayerWidget(QLabel):
    """
    Consumer Widget that receives frames and renders them on screen.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.setText("No Video Source Loaded")
        # Ensure it has a reasonable default size for the test
        self.setMinimumSize(640, 480)

    @Slot(np.ndarray)
    def receive_frame(self, frame: np.ndarray):
        """Slot to receive an OpenCV frame (BGR array), convert, and display it."""
        # Convert from OpenCV BGR to PySide RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, channels = rgb_frame.shape
        bytes_per_line = channels * w
        
        # Create QImage from numpy data
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and apply to QLabel with aspect ratio scaling
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    import sys

    class TestWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("RT-FFAT Module Test: VideoSource & VideoPlayerWidget")
            self.resize(800, 600)

            # 1. Initialize Modules
            self.video_source = VideoSource()
            self.player_widget = VideoPlayerWidget()

            # 2. Connect Signals & Slots (Producer -> Consumer)
            self.video_source.frame_ready.connect(self.player_widget.receive_frame)
            self.video_source.error_occurred.connect(self.show_error)

            # 3. Setup UI Layout
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(self.player_widget, stretch=1)

            btn_layout = QHBoxLayout()
            self.btn_load_cam = QPushButton("1. Load Camera")
            self.btn_load_video = QPushButton("2. Load Video File")
            
            # Combining Play/Pause into a toggle for a cleaner UI, or keeping separate for explicitness
            self.btn_play = QPushButton("3. Play")
            self.btn_pause = QPushButton("4. Pause")

            btn_layout.addWidget(self.btn_load_cam)
            btn_layout.addWidget(self.btn_load_video)
            btn_layout.addWidget(self.btn_play)
            btn_layout.addWidget(self.btn_pause)

            main_layout.addLayout(btn_layout)

            # 4. Connect Button Interactions
            self.btn_load_cam.clicked.connect(self.load_camera)
            self.btn_load_video.clicked.connect(self.load_video)
            self.btn_play.clicked.connect(self.video_source.play)
            self.btn_pause.clicked.connect(self.video_source.pause)

        def load_camera(self):
            self.btn_load_cam.setText("Detecting...")
            QApplication.processEvents() # Force UI update immediately
            
            cameras = VideoSource.get_camera_sources()
            if cameras:
                cam_id = cameras[0] # Grab the first available camera
                if self.video_source.load_source(cam_id):
                    self.video_source.play()
            else:
                self.show_error("No functional webcams detected.")
            self.btn_load_cam.setText("1. Load Camera")

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
            # Critical: Ensure thread shuts down properly when the window closes
            self.video_source.release()
            super().closeEvent(event)

    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
