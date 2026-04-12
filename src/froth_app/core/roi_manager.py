class ROICoordinateManager:
    """
    Core engine module that manages Region of Interest (ROI) coordinates.
    Decoupled from GUI elements to allow headless testing or background operations.
    """
    def __init__(self, max_rois=3):
        self.rois = [] # List of tuples: (x, y, w, h) in original image coordinates
        self.max_rois = max_rois

    def new_roi_coordinate(self, x, y, w, h, original_width, original_height):
        """
        Validates and adds a new ROI coordinate to the stack.
        Returns:
            (success_bool, status_message)
        """
        # Validate size: minimum 20x20 pixels
        if w < 20 or h < 20:
            return False, "ROI is too small or narrow. Please draw a larger box."
        
        # Enforce maximum
        if len(self.rois) >= self.max_rois:
            return False, f"Maximum of {self.max_rois} ROIs allowed."

        # Clamp boundaries to ensure we don't go outside the actual image
        x = max(0, min(int(x), original_width - 1))
        y = max(0, min(int(y), original_height - 1))
        w = max(1, min(int(w), original_width - x))
        h = max(1, min(int(h), original_height - y))

        self.rois.append((x, y, w, h))
        return True, "Successfully added ROI"

    def remove_last_roi(self):
        """Removes the most recently added ROI (LIFO behavior)."""
        if self.rois:
            self.rois.pop()
