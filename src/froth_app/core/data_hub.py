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
1. Add a _process_<algo>() method that returns a clean dict of meaningful fields.
2. Add a _format_<algo>() method that accepts that clean dict and returns a log str.
3. Register both in _PROCESSORS and _FORMATTERS at the bottom of the class.
4. Optionally add a Qt Signal and emit it inside _ingest().
"""

import math
import time
import numpy as np
from PySide6.QtCore import QThread, Signal
from queue import Empty
from multiprocessing import Queue
from froth_app.core.calibration import CalibrationManager
from froth_app.engine.algorithms.pca_handler import LBPPCAHandler


class GlobalDataHub(QThread):
    """
    Background thread that collects analysis results from all ROIWorker
    processes, groups them by ROI, and prints a combined report each time
    any algorithm for a given ROI produces a new result.

    Signals emit *processed* dicts — only the meaningful, UI-ready fields —
    rather than the raw worker payload.
    """

    # Emitted after LK result is processed. Payload is a clean dict:
    # { "dx_pixels", "dy_pixels", "real_distance", "unit", "features_tracked" }
    lk_data_ready = Signal(int, object)

    # Emitted after LBP result is processed. Payload is a clean dict:
    # { "texture_change_score", "hist_r", "hist_g", "hist_b", "features_tracked" }
    lbp_data_ready = Signal(int, object)

    def __init__(self, calibration_manager: CalibrationManager, parent=None):
        super().__init__(parent)
        self.collection_queue = Queue()
        self.calibration = calibration_manager
        self._is_running = False
        self.baseline_duration = 1.5

        # Latest *processed* result per ROI per algorithm:
        # { roi_id: { algo_name: processed_dict } }
        self._roi_buffer: dict[int, dict[str, dict]] = {}
        
        # LBP PCA State Handlers per ROI:
        # { roi_id: LBPPCAHandler }
        self._lbp_handlers: dict[int, LBPPCAHandler] = {}

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
        1. Extract the algorithm name and ROI id.
        2. Call the appropriate _process_* method to get a clean dict.
        3. Store the clean dict in the buffer.
        4. Emit the relevant Qt Signal with the clean dict.
        5. Reprint the combined log block for that ROI.
        """
        roi_id = raw_data.get("roi_id", 0)
        algo   = raw_data.get("algorithm", "Unknown")

        # --- Process raw → clean ---
        processor = self._PROCESSORS.get(algo, self._process_unknown)
        processed = processor(self, raw_data)

        # --- Store clean result ---
        if roi_id not in self._roi_buffer:
            self._roi_buffer[roi_id] = {}
        self._roi_buffer[roi_id][algo] = processed

        # --- Emit signal with clean dict (thread-safe via Qt queue) ---
        if algo == "LucasKanadeAlgorithm":
            self.lk_data_ready.emit(roi_id, processed)
        elif algo == "LBPAlgorithm":
            self.lbp_data_ready.emit(roi_id, processed)

        # --- Log ---
        self._print_roi(roi_id)

    def _print_roi(self, roi_id: int):
        """Print one combined block covering all algorithm results for the ROI."""
        algo_results = self._roi_buffer.get(roi_id, {})
        if not algo_results:
            return

        lines = [f"[Data Hub] ROI {roi_id + 1}"]
        for algo_name, processed in algo_results.items():
            formatter = self._FORMATTERS.get(algo_name, self._format_unknown)
            lines.append(f"  [{algo_name}] {formatter(self, processed)}")

        # print("\n".join(lines))

    # ------------------------------------------------------------------
    # Processors — raw worker dict → clean UI-ready dict
    # These are the single source of truth for what each signal carries.
    # ------------------------------------------------------------------

    def _process_lucas_kanade(self, raw: dict) -> dict:
        dx_p    = raw.get("dx_pixels", 0.0)
        dy_p    = raw.get("dy_pixels", 0.0)
        tracked = raw.get("features_tracked", 0)

        pixel_magnitude = math.sqrt(dx_p ** 2 + dy_p ** 2)
        real_distance   = self.calibration.get_real_distance(pixel_magnitude)

        return {
            "dx_pixels":       dx_p,
            "dy_pixels":       dy_p,
            "pixel_magnitude": pixel_magnitude,
            "real_distance":   real_distance,
            "unit":            self.calibration.unit_name,
            "features_tracked": tracked,
        }

    def _process_lbp(self, raw: dict) -> dict:
        roi_id = raw.get("roi_id", 0)
        hr = raw.get("lbp_r_hist")
        hg = raw.get("lbp_g_hist")
        hb = raw.get("lbp_b_hist")
        score = raw.get("texture_change_score", 0.0)
        tracked = raw.get("features_tracked", 0)

        # Base clean dict
        result = {
            "texture_change_score": score,
            "features_tracked": tracked,
            "is_baseline": True,
            "pc1": None,
            "pc2": None,
            "elapsed": 0.0
        }

        if hr is None or hg is None or hb is None:
            return result

        combined_hist = np.concatenate([hr, hg, hb])

        if roi_id not in self._lbp_handlers:
            self._lbp_handlers[roi_id] = LBPPCAHandler(baseline_duration=self.baseline_duration)

        # Delegate PCA logic to handler
        pca_result = self._lbp_handlers[roi_id].process_frame(combined_hist, roi_id)
        result.update(pca_result)

        return result

    def _process_unknown(self, raw: dict) -> dict:
        return dict(raw)  # pass-through, stripping nothing

    # ------------------------------------------------------------------
    # Formatters — clean dict → one-line log string
    # These consume the *processed* dict, not the raw worker payload.
    # ------------------------------------------------------------------

    def _format_lucas_kanade(self, processed: dict) -> str:
        return (
            f"Vector(x:{processed['dx_pixels']:.2f}, y:{processed['dy_pixels']:.2f}) px -> "
            f"Moved: {processed['real_distance']:.4f} {processed['unit']}/frame | "
            f"Tracked {processed['features_tracked']} elements"
        )

    def _format_lbp(self, processed: dict) -> str:
        base_str = (
            f"Texture change: {processed.get('texture_change_score', 0.0):.4f} | "
            f"Pixels: {processed.get('features_tracked', 0)}"
        )
        
        if processed.get("is_baseline", True):
            elapsed = processed.get("elapsed", 0.0)
            return f"{base_str} | [Baseline phase: {elapsed:.2f}s / {self.baseline_duration:.1f}s]"
        else:
            pc1 = processed.get("pc1", 0.0)
            pc2 = processed.get("pc2", 0.0)
            return f"{base_str} | PC1: {pc1:.4f}, PC2: {pc2:.4f}"

    def reset_roi_lbp_state(self, roi_id: int):
        """Clear the LBP baseline state for an ROI so it starts over."""
        if roi_id in self._lbp_handlers:
            del self._lbp_handlers[roi_id]

    def _format_unknown(self, processed: dict) -> str:
        return repr(processed)

    def update_baseline_duration(self, duration: float):
        self.baseline_duration = duration
        for handler in self._lbp_handlers.values():
            handler.baseline_duration = duration

    # ------------------------------------------------------------------
    # Registries — map algorithm class name → method
    # ------------------------------------------------------------------
    _PROCESSORS: dict = {
        "LucasKanadeAlgorithm": _process_lucas_kanade,
        "LBPAlgorithm":         _process_lbp,
    }

    _FORMATTERS: dict = {
        "LucasKanadeAlgorithm": _format_lucas_kanade,
        "LBPAlgorithm":         _format_lbp,
    }

    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean closure."""
        self._is_running = False
        self.wait()
