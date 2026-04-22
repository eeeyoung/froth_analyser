"""
calibration.py — CalibrationManager

Centralized engine to manage resolutions, real-world scale calibrations,
and overflow direction.

Overflow direction conventions
-------------------------------
Visual degrees (shown to the user):
    0°   = 12 o'clock  (straight up)
    90°  = 3  o'clock  (right)
    180° = 6  o'clock  (straight down)   ← default
    270° = 9  o'clock  (left)
    Increases clockwise.

Image-space axis vector (used by DataHub for projection):
    x: positive = right,  y: positive = DOWN  (screen/OpenCV convention)
    axis_x = sin(visual_rad)
    axis_y = -cos(visual_rad)
    Examples: 0°→(0,-1)  90°→(1,0)  180°→(0,1)  270°→(-1,0)
"""

import math


class CalibrationManager:
    """
    Shared calibration state for the entire application.
    All modules read from this single instance; only the UI writes to it.
    """

    def __init__(self):
        # --- Resolution ---
        self.raw_width = 1920
        self.raw_height = 1080
        self.processing_width = 800
        self.processing_height = 600

        # --- Scale conversion ---
        self.pixels_per_unit = 1.0
        self.unit_name = "mm"

        # --- Overflow direction ---
        # Default: 180° = 6 o'clock = straight downward into launder
        self._overflow_direction_visual: float = 180.0
        self.overflow_calibrated: bool = False

    # ------------------------------------------------------------------
    # Overflow direction API
    # ------------------------------------------------------------------

    @property
    def overflow_direction_visual(self) -> float:
        """Current overflow direction in visual degrees [0, 360)."""
        return self._overflow_direction_visual

    def get_overflow_axis_image(self) -> tuple[float, float]:
        """
        Returns the overflow unit vector in image (screen) coordinates.
            axis_x: positive = right
            axis_y: positive = DOWN  (image convention)

        Projection of LK displacement onto overflow:
            projected = dx_pixels * axis_x + dy_pixels * axis_y
            Positive  → bubble moving WITH the overflow
            Negative  → bubble moving AGAINST the overflow
        """
        rad = math.radians(self._overflow_direction_visual)
        return math.sin(rad), -math.cos(rad)

    def set_overflow_visual(self, deg: float) -> None:
        """Set overflow direction in visual degrees. Does NOT set calibrated flag."""
        self._overflow_direction_visual = float(deg) % 360.0

    def confirm_overflow(self) -> None:
        """Mark the overflow direction as confirmed by the user."""
        self.overflow_calibrated = True

    def reset_overflow(self) -> None:
        """Unconfirm calibration (keeps last angle so the widget shows it again)."""
        self.overflow_calibrated = False

    # ------------------------------------------------------------------
    # Resolution API
    # ------------------------------------------------------------------

    def update_raw_resolution(self, width: int, height: int) -> None:
        self.raw_width = max(1, int(width))
        self.raw_height = max(1, int(height))

    def update_processing_resolution(self, width: int, height: int) -> None:
        self.processing_width = max(1, int(width))
        self.processing_height = max(1, int(height))

    # ------------------------------------------------------------------
    # Scale conversion API
    # ------------------------------------------------------------------

    def update_conversion_rate(
        self, num_pixels: float, real_distance: float, unit_name: str = "mm"
    ):
        if real_distance <= 0 or num_pixels <= 0:
            return False, "Values must be greater than zero."
        self.pixels_per_unit = num_pixels / real_distance
        self.unit_name = unit_name
        return True, f"Success: 1 {self.unit_name} = {self.pixels_per_unit:.2f} pixels."

    def get_real_distance(self, pixels: float) -> float:
        return pixels / self.pixels_per_unit
