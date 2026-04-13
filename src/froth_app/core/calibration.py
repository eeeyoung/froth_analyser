class CalibrationManager:
    """
    Centralized engine to manage resolutions and real-world scale calibrations.
    Shared across the application to ensure accurate math in UI and analysis.
    """
    def __init__(self):
        # 1. Input raw resolution (from camera/video)
        self.raw_width = 1920
        self.raw_height = 1080
        
        # 2. Processing resolution (for the Analysis Engine)
        self.processing_width = 800
        self.processing_height = 600
        
        # 3. Conversion rate
        self.pixels_per_unit = 1.0  
        self.unit_name = "mm"

    def update_raw_resolution(self, width: int, height: int):
        """Updates the native source resolution."""
        self.raw_width = max(1, int(width))
        self.raw_height = max(1, int(height))

    def update_processing_resolution(self, width: int, height: int):
        """Updates the target downscaled resolution for heavy OpenCV processing."""
        self.processing_width = max(1, int(width))
        self.processing_height = max(1, int(height))

    def update_conversion_rate(self, num_pixels: float, real_distance: float, unit_name: str = "mm"):
        """
        Calculates and stores the conversion factor.
        Example: User draws a line over a pipe covering 150 pixels, 
        and inputs that the pipe is 10 mm wide in real life.
        """
        if real_distance <= 0 or num_pixels <= 0:
            return False, "Values must be greater than zero."
            
        self.pixels_per_unit = num_pixels / real_distance
        self.unit_name = unit_name
        return True, f"Success: 1 {self.unit_name} = {self.pixels_per_unit:.2f} pixels."

    def get_real_distance(self, pixels: float) -> float:
        """Helper to safely convert pixels to real-world units."""
        return pixels / self.pixels_per_unit
