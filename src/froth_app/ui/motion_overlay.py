"""
motion_overlay.py — MotionCrosshairOverlay

A transparent QWidget stacked on top of a CroppedROIWidget that draws a
full-span crosshair confined to the actual video image area (respecting the
letterbox margins from Qt.KeepAspectRatio scaling).

The crosshair intersection point moves with each Lucas-Kanade (dx, dy) frame
result and wraps toroidally — reaching one edge makes it reappear from the
opposite edge, creating a continuous Pac-Man style motion.

Integration
-----------
1. Pass the matching CroppedROIWidget reference to the constructor.
2. Stack both in a QGridLayout at (0, 0) inside a container QWidget.
3. Connect DataHub.lk_data_ready → dispatch to overlay.update_motion(dx, dy).
4. Toggle visibility with overlay.setVisible(bool).
"""

from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Qt, QPointF, QRect
from PySide6.QtGui import QPainter, QPen, QColor


class MotionCrosshairOverlay(QWidget):
    """
    Transparent overlay that renders a moving full-span crosshair, restricted
    to the exact region of the displayed video inside a CroppedROIWidget.

    Position convention (fractional, within the image rect):
        _cx = 0.5  →  horizontal centre
        _cy = 0.5  →  vertical centre
        _cx = 0.0 / 1.0  →  left / right edge (wraps)
        _cy = 0.0 / 1.0  →  top / bottom edge (wraps)
    """

    # Screen multiplier: how many overlay pixels shift per flow pixel.
    _SCALE = 5.0

    def __init__(self, crop_widget: QLabel, parent: QWidget | None = None):
        """
        Parameters
        ----------
        crop_widget : QLabel (CroppedROIWidget)
            The label whose displayed pixmap defines the drawing region.
            The overlay must be stacked in the same grid cell so their
            coordinate origins coincide.
        """
        super().__init__(parent)
        self._crop_widget = crop_widget

        # Fully transparent — passes all mouse events through
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        # Crosshair position as a fraction [0.0, 1.0) of the image rect
        self._cx: float = 0.5
        self._cy: float = 0.5

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _image_rect(self) -> QRect:
        """
        Compute the sub-rectangle of this overlay that corresponds to the
        actual displayed video frame inside the CroppedROIWidget.

        CroppedROIWidget scales with Qt.KeepAspectRatio and Qt.AlignCenter,
        so the pixmap is centred with equal letterbox margins on each axis.

        Falls back to the full overlay rect if no pixmap is set yet.
        """
        pixmap = self._crop_widget.pixmap()
        if pixmap is None or pixmap.isNull():
            return self.rect()

        pw = pixmap.width()
        ph = pixmap.height()
        lw = self._crop_widget.width()
        lh = self._crop_widget.height()

        x_off = (lw - pw) // 2
        y_off = (lh - ph) // 2

        return QRect(x_off, y_off, pw, ph)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_motion(self, dx: float, dy: float) -> None:
        """
        Feed one frame's (dx, dy) displacement in pixels and repaint.
        Converts to fractions using the displayed image rect dimensions,
        then delegates to push_fraction().
        """
        rect = self._image_rect()
        w = rect.width()
        h = rect.height()
        if w == 0 or h == 0:
            return
        self.push_fraction((dx * self._SCALE) / w, (dy * self._SCALE) / h)

    def push_fraction(self, frac_dx: float, frac_dy: float) -> None:
        """
        Apply a pre-calculated fractional shift directly.

        Use this from ROIDetailWindow, which knows the original crop dimensions
        and converts LK pixel displacement to fractions in crop space:
            frac_dx = dx_pixels / crop_original_width

        Parameters
        ----------
        frac_dx : float   Fractional horizontal shift (fraction of image width).
        frac_dy : float   Fractional vertical shift (fraction of image height).
        """
        self._cx = (self._cx + frac_dx) % 1.0
        self._cy = (self._cy + frac_dy) % 1.0

        self.update()

    def reset(self) -> None:
        """Snap crosshair back to centre (call when the ROI is removed)."""
        self._cx = 0.5
        self._cy = 0.5
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)

        rect = self._image_rect()
        if rect.width() == 0 or rect.height() == 0:
            return

        # Absolute pixel position of the crosshair intersection
        cx = rect.x() + self._cx * rect.width()
        cy = rect.y() + self._cy * rect.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Clip all drawing strictly to the video image area
        painter.setClipRect(rect)

        # --- Crosshair lines ---
        line_pen = QPen(QColor(0, 220, 255, 200))   # cyan, semi-transparent
        line_pen.setWidth(1)
        painter.setPen(line_pen)

        # Horizontal line — spans full image width at height cy
        painter.drawLine(
            QPointF(rect.left(),  cy),
            QPointF(rect.right(), cy),
        )
        # Vertical line — spans full image height at position cx
        painter.drawLine(
            QPointF(cx, rect.top()),
            QPointF(cx, rect.bottom()),
        )

        # --- Centre dot ---
        dot_pen = QPen(QColor(255, 255, 255, 240))
        dot_pen.setWidth(3)
        painter.setPen(dot_pen)
        painter.drawPoint(QPointF(cx, cy))

        painter.end()
