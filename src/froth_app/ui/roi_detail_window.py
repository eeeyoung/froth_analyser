"""
roi_detail_window.py — ROIDetailWindow

A standalone popup window that opens when the user clicks a CroppedROIWidget
thumbnail. It displays the ROI crop at its original resolution (1 : 1 pixels,
or scaled down to fit a maximum window size) and optionally overlays a
MotionCrosshairOverlay driven by Lucas-Kanade displacement data.

Key behaviours
--------------
- Window is non-modal — the main window stays responsive while it is open.
- Updated every video frame via update_frame(crop_np).
- Receives LK data via update_lk(dx, dy); converts from crop-pixel space to
  fractional coordinates using the known crop dimensions so the crosshair
  position is physically accurate at any display scale.
- Crosshair is only visible when LK is active (controlled by set_lk_visible).
- Closing the popup just hides it; re-clicking the thumbnail shows it again.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QCloseEvent

from froth_app.ui.motion_overlay import MotionCrosshairOverlay


# Maximum pixel dimension (width or height) of the display label.
# Crops larger than this are scaled down; smaller crops are shown 1 : 1.
_MAX_DIM = 720


class ROIDetailWindow(QWidget):
    """
    Popup live-view window for a single ROI.

    Parameters
    ----------
    roi_index : int
        0-based ROI index, used only for the window title.
    lk_active : bool
        Initial visibility of the motion crosshair overlay.
    """

    def __init__(self, roi_index: int, lk_active: bool, parent: QWidget | None = None):
        # Qt.Window — independent top-level window; Qt.Tool keeps it on top of
        # the parent without appearing in the taskbar (optional, change to
        # Qt.Window for a regular window).
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"ROI {roi_index + 1} — Live View")
        self.setAttribute(Qt.WA_DeleteOnClose, False)   # hide, don't destroy

        # Track original crop dimensions for accurate fraction conversion
        self._crop_w: int = 1
        self._crop_h: int = 1

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container holds the label and crosshair stacked in the same cell
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background-color: #0d0d0d;")

        # Crosshair overlay — passes the label so it can compute the image rect
        self._crosshair = MotionCrosshairOverlay(crop_widget=self._label)
        self._crosshair.setVisible(lk_active)

        grid.addWidget(self._label,     0, 0)
        grid.addWidget(self._crosshair, 0, 0)

        layout.addWidget(container)

    # ------------------------------------------------------------------
    # Public API — called by the main window
    # ------------------------------------------------------------------

    def update_frame(self, crop: np.ndarray) -> None:
        """
        Display a new crop frame.  Called on every video frame.

        The crop is shown at its native resolution (1 : 1) unless it exceeds
        _MAX_DIM in either dimension, in which case it is scaled down
        proportionally.  The window is resized to match on the first frame and
        whenever the ROI dimensions change.

        Parameters
        ----------
        crop : np.ndarray  BGR image from OpenCV in the original ROI dimensions.
        """
        if crop is None or crop.size == 0:
            return

        h, w = crop.shape[:2]

        # Resize window only when dimensions change (avoids jitter every frame)
        if w != self._crop_w or h != self._crop_h:
            self._crop_w = w
            self._crop_h = h
            scale = min(_MAX_DIM / w, _MAX_DIM / h, 1.0)   # never upscale
            display_w = max(1, int(w * scale))
            display_h = max(1, int(h * scale))
            self._label.setFixedSize(display_w, display_h)
            self._crosshair.setFixedSize(display_w, display_h)
            self.adjustSize()

        # Convert BGR → RGB and build QPixmap
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        bytes_per_line = rgb.shape[1] * 3
        q_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to the fixed label size (KeepAspectRatio for non-square labels)
        self._label.setPixmap(
            pixmap.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_lk(self, dx: float, dy: float) -> None:
        """
        Forward LK pixel displacement to the crosshair using crop-space fractions.

        LK computes displacement in the original crop coordinate space, so
        dividing by the crop dimensions gives the correct fractional shift
        regardless of the display scale factor.

        Parameters
        ----------
        dx : float  Horizontal displacement in **crop** pixels.
        dy : float  Vertical   displacement in **crop** pixels.
        """
        if self._crop_w == 0 or self._crop_h == 0:
            return
        print(f'dx: {dx}, dy: {dy}, crop_w: {self._crop_w}, crop_h: {self._crop_h}')
        self._crosshair.push_fraction(dx / self._crop_w, dy / self._crop_h)

    def set_lk_visible(self, visible: bool) -> None:
        """Show or hide the motion crosshair overlay."""
        self._crosshair.setVisible(visible)

    def reset(self) -> None:
        """Snap crosshair to centre and clear the frame (called on ROI removal)."""
        self._crosshair.reset()
        self._label.clear()
        self._label.setText("ROI removed")

    # ------------------------------------------------------------------
    # Override close to hide rather than destroy
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hide on close so re-clicking the thumbnail restores the window."""
        self.hide()
        event.ignore()
