"""
data_hub.py — GlobalDataHub

Singleton-style background QThread that listens to the shared IPC collection
queue where all ROIWorker processes drop their results.

Each result dict contains an 'algorithm' key (injected by ROIWorker). Because
a single ROI can run multiple algorithms in parallel, results arrive separately
per algorithm. The DataHub accumulates the latest result from each algorithm
per ROI, and prints a combined ROI report every time any algorithm updates.

Adding support for a new algorithm
------------------------------------
1. Add a handler method _format_<algo_name>() that returns a one-line string.
2. Register it in _FORMATTERS at the bottom of the class.
"""

import math
from PySide6.QtCore import QThread, Signal
from queue import Empty
from multiprocessing import Queue
from froth_app.core.calibration import CalibrationManager


class GlobalDataHub(QThread):
    """
    Background thread that collects analysis results from all ROIWorker
    processes, groups them by ROI, and prints a combined report each time
    any algorithm for a given ROI produces a new result.
    """

    # Emitted on the main thread whenever an LBP result arrives.
    # Args: (roi_id: int, result: dict) — connect to UI plot widgets.
    lbp_data_ready = Signal(int, object)

    def __init__(self, calibration_manager: CalibrationManager, parent=None):
        super().__init__(parent)
        # IPC Queue that multiple Worker Processes will write to
        self.collection_queue = Queue()
        self.calibration = calibration_manager
        self._is_running = False

        # Latest result per ROI per algorithm:
        # { roi_id: { algo_name: result_dict } }
        self._roi_buffer: dict[int, dict[str, dict]] = {}

    def run(self):
        self._is_running = True
        while self._is_running:
            try:
                raw_data = self.collection_queue.get(timeout=0.1)
                self._ingest(raw_data)
            except Empty:
                continue
            except Exception as e:
                print(f"[Data Hub] Error: {e}")

    def _ingest(self, raw_data: dict):
        """
        Store the incoming result in the per-ROI buffer under its algorithm
        name, then reprint the full combined report for that ROI.
        """
        roi_id = raw_data.get("roi_id", 0)
        algo   = raw_data.get("algorithm", "Unknown")
        print(f"[Data Hub] Ingesting {algo} for ROI {roi_id + 1}")
        # Update the buffer slot for this (roi_id, algorithm) pair
        if roi_id not in self._roi_buffer:
            self._roi_buffer[roi_id] = {}
        self._roi_buffer[roi_id][algo] = raw_data

        # Signal the UI for live chart updates (crosses thread safely via Qt queue)
        if algo == "LBPAlgorithm":
            self.lbp_data_ready.emit(roi_id, raw_data)

        # Reprint the full combined block for this ROI
        self._print_roi(roi_id)

    def _print_roi(self, roi_id: int):
        """Print one combined block covering all algorithm results for the ROI."""
        algo_results = self._roi_buffer.get(roi_id, {})
        if not algo_results:
            return

        lines = [f"[Data Hub] ROI {roi_id + 1}"]
        for algo_name, result in algo_results.items():
            formatter = self._FORMATTERS.get(algo_name, self._format_unknown)
            lines.append(f"  [{algo_name}] {formatter(self, result)}")

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Per-algorithm line formatters
    # Each returns a single descriptive string for its result dict.
    # ------------------------------------------------------------------

    def _format_lucas_kanade(self, result: dict) -> str:
        dx_p    = result.get("dx_pixels", 0.0)
        dy_p    = result.get("dy_pixels", 0.0)
        tracked = result.get("features_tracked", 0)

        pixel_magnitude = math.sqrt(dx_p ** 2 + dy_p ** 2)
        real_distance   = self.calibration.get_real_distance(pixel_magnitude)
        unit            = self.calibration.unit_name

        return (
            f"Vector(x:{dx_p:.2f}, y:{dy_p:.2f}) px -> "
            f"Moved: {real_distance:.4f} {unit}/frame | "
            f"Tracked {tracked} elements"
        )

    def _format_lbp(self, result: dict) -> str:
        score   = result.get("texture_change_score", 0.0)
        tracked = result.get("features_tracked", 0)
        hist_r  = result.get("lbp_r_hist")
        hist_g  = result.get("lbp_g_hist")
        hist_b  = result.get("lbp_b_hist")

        peak_r = int(hist_r.argmax()) if hist_r is not None else -1
        peak_g = int(hist_g.argmax()) if hist_g is not None else -1
        peak_b = int(hist_b.argmax()) if hist_b is not None else -1

        return (
            f"Texture change: {score:.4f} | "
            f"Peak codes R:{peak_r} G:{peak_g} B:{peak_b} | "
            f"Pixels: {tracked}"
        )

    def _format_unknown(self, result: dict) -> str:
        return repr(result)

    # ------------------------------------------------------------------
    # Formatter registry — maps algorithm class name -> formatter method
    # ------------------------------------------------------------------
    _FORMATTERS: dict = {
        "LucasKanadeAlgorithm": _format_lucas_kanade,
        "LBPAlgorithm":         _format_lbp,
    }

    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean closure."""
        self._is_running = False
        self.wait()
