"""
overflow_calibration_widget.py — Compass-rose overlay for overflow direction.

Stacked on top of the video canvas (same QGridLayout slot as ROIOverlayWidget).
The user drags the golden needle or clicks a cardinal label to set the froth
overflow direction. Updates CalibrationManager live; confirmation is handled
by the parent button.

Visual degree convention
------------------------
    0°   = 12 o'clock (up)     90°  = 3 o'clock (right)
    180° = 6 o'clock (down)    270° = 9 o'clock (left)
    Increases clockwise.
"""

import math
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QFontMetrics,
    QRadialGradient, QPainterPath,
)
from PySide6.QtWidgets import QWidget

from froth_app.core.calibration import CalibrationManager

# (visual_degrees, label, is_cardinal)
_COMPASS_LABELS = [
    (0,   "0",  True),  (45,  "45", False), (90,  "90",  True),
    (135, "135", False), (180, "180",  True),   (225, "225", False),
    (270, "270",  True),  (315, "315", False),
]

_DIRECTION_NAMES = {
    0: "Up", 45: "Up-Right", 90: "Right",
    135: "Down-Right", 180: "Down", 225: "Down-Left",
    270: "Left", 315: "Up-Left",
}


def _nearest_name(angle: float) -> str:
    nearest = min(_DIRECTION_NAMES, key=lambda k: abs((angle - k + 180) % 360 - 180))
    return _DIRECTION_NAMES[nearest]


class OverflowCalibrationWidget(QWidget):
    """
    Transparent overlay widget that renders a compass-rose dial for
    overflow direction calibration. Hide/show via activate()/deactivate().
    """

    direction_changed = Signal(float)   # emits visual degrees on every change

    # ------------------------------------------------------------------ #
    def __init__(self, calibration: CalibrationManager, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent;")

        self._active = False
        self._dragging = False
        self._current_angle: float = calibration.overflow_direction_visual
        self.hide()

    # ------------------------------------------------------------------ #
    # Activation
    # ------------------------------------------------------------------ #

    def activate(self):
        self._current_angle = self.calibration.overflow_direction_visual
        self._active = True
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.show()
        self.update()

    def deactivate(self):
        self._active = False
        self._dragging = False
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def _cx(self) -> float: return self.width()  / 2.0
    def _cy(self) -> float: return self.height() / 2.0
    def _cr(self) -> float: return min(self.width(), self.height()) * 0.27   # label ring radius
    def _nr(self) -> float: return self._cr() * 0.70                          # needle length
    def _hr(self) -> float: return 15.0                                        # drag handle radius

    def _tip(self, angle: float) -> QPointF:
        r, rad = self._nr(), math.radians(angle)
        return QPointF(self._cx() + r * math.sin(rad), self._cy() - r * math.cos(rad))

    def _label_center(self, angle: float) -> QPointF:
        r, rad = self._cr(), math.radians(angle)
        return QPointF(self._cx() + r * math.sin(rad), self._cy() - r * math.cos(rad))

    @staticmethod
    def _dist(a: QPointF, b: QPointF) -> float:
        return math.hypot(a.x() - b.x(), a.y() - b.y())

    def _cursor_angle(self, pos: QPointF) -> float:
        dx, dy = pos.x() - self._cx(), pos.y() - self._cy()
        return math.degrees(math.atan2(dx, -dy)) % 360.0

    def _set_angle(self, deg: float):
        self._current_angle = deg % 360.0
        self.calibration.set_overflow_visual(self._current_angle)
        self.direction_changed.emit(self._current_angle)
        self.update()

    # ------------------------------------------------------------------ #
    # Mouse events
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event):
        if not self._active:
            return
        pos = event.position()

        # Click on a compass label → snap
        for ang, _, _ in _COMPASS_LABELS:
            if self._dist(pos, self._label_center(ang)) <= 26:
                self._set_angle(float(ang))
                return

        # Click near needle tip → start drag
        if self._dist(pos, self._tip(self._current_angle)) <= self._hr() + 10:
            self._dragging = True

    def mouseMoveEvent(self, event):
        if self._active and self._dragging:
            self._set_angle(self._cursor_angle(event.position()))

    def mouseReleaseEvent(self, event):
        self._dragging = False

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #

    def paintEvent(self, event):
        if not self._active:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw_backdrop(p)
        self._draw_ticks(p)
        self._draw_labels(p)
        self._draw_needle(p)
        self._draw_info(p)
        p.end()

    # --- backdrop -------------------------------------------------------

    def _draw_backdrop(self, p: QPainter):
        p.fillRect(self.rect(), QColor(0, 0, 0, 150))

        cx, cy, cr = self._cx(), self._cy(), self._cr()
        grad = QRadialGradient(QPointF(cx, cy), cr * 1.30)
        grad.setColorAt(0.0, QColor(15, 20, 45, 230))
        grad.setColorAt(0.75, QColor(10, 12, 30, 200))
        grad.setColorAt(1.0,  QColor(0,  0,  0,  0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), cr * 1.28, cr * 1.28)

        # Decorative outer ring
        p.setPen(QPen(QColor(70, 130, 200, 100), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), cr * 1.10, cr * 1.10)

    # --- tick marks -----------------------------------------------------

    def _draw_ticks(self, p: QPainter):
        cx, cy = self._cx(), self._cy()
        r_out = self._cr() * 0.90
        for deg in range(0, 360, 10):
            cardinal = deg % 90 == 0
            intercard = deg % 45 == 0 and not cardinal
            r_in = r_out * (0.86 if cardinal else 0.90 if intercard else 0.94)
            width = 1.8 if cardinal else 1.3 if intercard else 0.9
            alpha = 180 if cardinal else 130 if intercard else 70
            pen = QPen(QColor(100, 180, 255, alpha), width)
            rad = math.radians(deg)
            s, c = math.sin(rad), math.cos(rad)
            p.setPen(pen)
            p.drawLine(
                QPointF(cx + r_in * s,  cy - r_in * c),
                QPointF(cx + r_out * s, cy - r_out * c),
            )

    # --- compass labels -------------------------------------------------

    def _draw_labels(self, p: QPainter):
        for ang, label, cardinal in _COMPASS_LABELS:
            lc = self._label_center(ang)
            r = 22 if cardinal else 18

            # Is needle near this label?
            diff = abs((self._current_angle - ang + 180) % 360 - 180)
            highlighted = diff < 22.5

            fill  = QColor(0, 170, 255, 240) if highlighted else (
                    QColor(35, 85, 155, 220) if cardinal else QColor(20, 50, 95, 180))
            border = QColor(80, 180, 255, 200)

            p.setBrush(QBrush(fill))
            p.setPen(QPen(border, 1.5))
            p.drawEllipse(lc, r, r)

            font = QFont("Segoe UI", 9 if cardinal else 8,
                         QFont.Bold if cardinal else QFont.Normal)
            p.setFont(font)
            p.setPen(QColor(220, 240, 255))
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(label)
            p.drawText(QPointF(lc.x() - tw / 2, lc.y() + fm.ascent() / 2 - 1), label)

    # --- needle ---------------------------------------------------------

    def _draw_needle(self, p: QPainter):
        cx, cy = self._cx(), self._cy()
        origin = QPointF(cx, cy)
        tip    = self._tip(self._current_angle)
        gold   = QColor(255, 205, 50)

        # Shadow
        off = QPointF(2, 2)
        p.setPen(QPen(QColor(0, 0, 0, 90), 5))
        p.drawLine(origin + off, tip + off)

        # Shaft
        p.setPen(QPen(gold, 3))
        p.drawLine(origin, tip)

        # Arrowhead
        dx, dy = tip.x() - cx, tip.y() - cy
        length = math.hypot(dx, dy)
        if length > 1:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            sz, hw = 16, 7
            bx, by = tip.x() - ux * sz, tip.y() - uy * sz
            path = QPainterPath()
            path.moveTo(tip)
            path.lineTo(bx + px * hw, by + py * hw)
            path.lineTo(bx - px * hw, by - py * hw)
            path.closeSubpath()
            p.setBrush(QBrush(gold))
            p.setPen(Qt.NoPen)
            p.drawPath(path)

        # Drag handle circle
        p.setBrush(QBrush(QColor(255, 220, 80, 210)))
        p.setPen(QPen(QColor(255, 255, 200), 2))
        p.drawEllipse(tip, self._hr(), self._hr())

        # Pivot dot
        p.setBrush(QBrush(QColor(255, 220, 80)))
        p.setPen(QPen(QColor(255, 255, 200), 1.5))
        p.drawEllipse(origin, 7, 7)

    # --- centre info text -----------------------------------------------

    def _draw_info(self, p: QPainter):
        cx, cy = self._cx(), self._cy()
        angle  = self._current_angle
        name   = _nearest_name(angle)

        # Angle value
        font_big = QFont("Segoe UI", 20, QFont.Bold)
        p.setFont(font_big)
        p.setPen(QColor(255, 220, 80))
        text_big = f"{angle:.0f}°"
        fm = QFontMetrics(font_big)
        p.drawText(QPointF(cx - fm.horizontalAdvance(text_big) / 2, cy - 10), text_big)

        # Direction name
        font_sm = QFont("Segoe UI", 9)
        p.setFont(font_sm)
        p.setPen(QColor(180, 210, 255, 200))
        fm_sm = QFontMetrics(font_sm)
        p.drawText(QPointF(cx - fm_sm.horizontalAdvance(name) / 2, cy + 14), name)

        # Instruction hint
        hint = "Drag the needle tip  or  click a label"
        font_hint = QFont("Segoe UI", 8)
        p.setFont(font_hint)
        p.setPen(QColor(120, 150, 200, 160))
        fm_h = QFontMetrics(font_hint)
        p.drawText(QPointF(cx - fm_h.horizontalAdvance(hint) / 2,
                           cy + self._cr() * 1.30), hint)
