import math
from PySide6.QtCore import QThread
from queue import Empty
from multiprocessing import Queue
from froth_app.core.calibration import CalibrationManager

class GlobalDataHub(QThread):
    """
    Singleton-style background thread that constantly listens to the 
    collection queue where all worker processes drop their raw pixel results.
    It applies physical calibration conversions dynamically.
    """
    def __init__(self, calibration_manager: CalibrationManager, parent=None):
        super().__init__(parent)
        # IPC Queue that multiple Worker Processes will write to
        self.collection_queue = Queue()
        self.calibration = calibration_manager
        self._is_running = False
        
    def run(self):
        self._is_running = True
        while self._is_running:
            try:
                # 1. Capture raw data from any Worker exactly as it arrives
                raw_data = self.collection_queue.get(timeout=0.1)
                
                roi_id = raw_data.get("roi_id")
                dx_p = raw_data.get("dx_pixels", 0)
                dy_p = raw_data.get("dy_pixels", 0)
                tracked_events = raw_data.get("features_tracked", 0)
                
                # 2. Conversion Magic using Pythagorean theorem
                pixel_magnitude = math.sqrt(dx_p**2 + dy_p**2)
                
                # Convert absolute pixel movement to millimeters (or configured unit)
                real_distance = self.calibration.get_real_distance(pixel_magnitude)
                unit = self.calibration.unit_name
                
                # Print terminal output as requested
                # E.g. [Data Hub] ROI 1: Moved 0.450 mm/frame (Tracking 43 bubbles)
                print(f"[Data Hub] ROI {roi_id+1} | Vector(x:{dx_p:.2f}, y:{dy_p:.2f}) px -> "
                      f"Moved: {real_distance:.4f} {unit}/frame | Tracked {tracked_events} elements")
                
            except Empty:
                continue # Loop back if no data arrived in 100ms
            except Exception as e:
                print(f"Data Hub Error: {str(e)}")
                
    def shutdown(self):
        """Clean closure."""
        self._is_running = False
        self.wait()
