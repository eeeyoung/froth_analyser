"""
analyzer.py — ROIWorker and AnalysisEngineMaster

Algorithm Registry
------------------
ALGORITHM_REGISTRY maps integer IDs to algorithm classes.
To register a new algorithm, add an entry here.

    1  → LucasKanadeAlgorithm
    (future entries go here)

ROIWorker
---------
An independent multiprocessing.Process that runs a *list* of
BaseAnalysisAlgorithm instances on a dedicated CPU core for one ROI stream.
Multiple algorithms can be loaded before the process starts via
add_new_algorithm(); all of them run sequentially on every incoming frame
and each posts its own result to the DataHub.

AnalysisEngineMaster
--------------------
Main-thread manager. Accepts a list of algorithm IDs per ROI, resolves
them through ALGORITHM_REGISTRY, spawns an ROIWorker, and loads it up
before starting the process. Handles routing and lifecycle management.

    Example — run LK (id=1) on ROI 0, and two algorithms on ROI 1:
        master.add_roi_stream(roi_id=0, algorithm_ids=[1])
        master.add_roi_stream(roi_id=1, algorithm_ids=[1, 2])
"""

import time
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty

from froth_app.engine.algorithms.base import BaseAnalysisAlgorithm
from froth_app.engine.algorithms.lucas_kanade import LucasKanadeAlgorithm
from froth_app.engine.algorithms.lbp import LBPAlgorithm


# ---------------------------------------------------------------------------
# Algorithm Registry
# ---------------------------------------------------------------------------
# Maps integer algorithm IDs (chosen by the user) to their algorithm classes.
# Add new algorithms here as lower-level modules are developed.
ALGORITHM_REGISTRY: dict[int, type[BaseAnalysisAlgorithm]] = {
    1: LucasKanadeAlgorithm,
    2: LBPAlgorithm,
}


class ROIWorker(Process):
    """
    Independent background process dedicated to analysing a single ROI stream.
    Runs on a separate CPU core (true parallelism via multiprocessing).

    Holds a list of BaseAnalysisAlgorithm instances. On every incoming frame
    each algorithm is called in turn and its result is forwarded to the
    DataHub independently, so consumers can distinguish algorithm outputs by
    the 'algorithm' key in the result dict.

    IMPORTANT: add_new_algorithm() must be called BEFORE start(). After the
    process is forked, parent-side mutations are not visible to the child.
    """

    def __init__(
        self,
        roi_id: int,
        input_queue: Queue,
        output_queue: Queue,
    ):
        super().__init__()
        self.roi_id = roi_id
        self.algorithms: list[BaseAnalysisAlgorithm] = []
        self.input_queue = input_queue
        self.output_queue = output_queue

    def add_new_algorithm(self, algorithm: BaseAnalysisAlgorithm) -> None:
        """
        Attach an algorithm to this worker.
        Must be called before start() — multiprocessing copies state at fork time.
        """
        self.algorithms.append(algorithm)

    def run(self):
        """Main execution loop — pure routing, zero algorithm logic here."""
        algo_names = [type(a).__name__ for a in self.algorithms]
        print(
            f"[Worker {self.roi_id + 1} | {', '.join(algo_names) or 'no algorithms'}]"
            f" Process booted successfully."
        )

        while True:
            try:
                # 1. Grab the next crop from the pipe (timeout allows clean exit)
                frame = self.input_queue.get(timeout=1.0)

                # Poison pill — shut down cleanly
                if frame is None:
                    break

                # 2. Run every loaded algorithm on the same frame
                for algorithm in self.algorithms:
                    result = algorithm.process_frame(frame)

                    # 3. Annotate and forward each result to the DataHub
                    if result is not None:
                        result["roi_id"] = self.roi_id
                        result["timestamp"] = time.time()
                        result["algorithm"] = type(algorithm).__name__
                        try:
                            self.output_queue.put_nowait(result)
                        except Exception:
                            pass  # Drop gracefully if the DataHub queue is full

            except Empty:
                continue  # Normal timeout; loop back and keep waiting
            except Exception as e:
                print(f"[Worker {self.roi_id + 1}] Crashed: {e}")
                break

        print(f"[Worker {self.roi_id + 1}] Shutting down safely.")


class AnalysisEngineMaster:
    """
    Manager running on the main thread.
    Spawns/terminates ROIWorker processes and routes video frame crops
    to them. Performs no computation itself.

    Algorithm selection
    -------------------
    add_roi_stream() accepts a list of integer algorithm IDs. Each ID is
    resolved through ALGORITHM_REGISTRY to an algorithm instance and loaded
    into the worker via add_new_algorithm() before the process is started.

    If algorithm_ids is empty or omitted, LucasKanade (id=1) is used as
    the default so existing call-sites require no changes.

    Multi-algorithm parallel example
    ---------------------------------
        master.add_roi_stream(roi_id=0, algorithm_ids=[1])
        master.add_roi_stream(roi_id=1, algorithm_ids=[1, 2])
    """

    def __init__(self, data_hub):
        # Maps roi_id -> {"process": ROIWorker, "input_q": Queue}
        self.workers: dict = {}
        self.data_hub = data_hub

    def add_roi_stream(self, roi_id: int, algorithm_ids: list[int] | None = None):
        """
        Spawn a new parallel worker process for a newly drawn ROI.

        Parameters
        ----------
        roi_id : int
            Unique identifier for the ROI (matches index in roi_list).
        algorithm_ids : list[int], optional
            List of algorithm IDs to run on this ROI. Each ID must exist in
            ALGORITHM_REGISTRY. Defaults to [1] (LucasKanade) if not provided.
        """
        if roi_id in self.workers:
            return

        if not algorithm_ids:
            algorithm_ids = [1]  # Default: LucasKanade

        # Bounded queue — acts as a frame-drop throttle if the worker is slow
        input_queue = Queue(maxsize=15)

        # Create the worker (no algorithms yet)
        worker = ROIWorker(roi_id, input_queue, self.data_hub.collection_queue)

        # Resolve each algorithm ID and load it into the worker before forking
        for algo_id in algorithm_ids:
            algo_class = ALGORITHM_REGISTRY.get(algo_id)
            if algo_class is None:
                print(
                    f"[AnalysisEngineMaster] Warning: algorithm id={algo_id} is not "
                    f"registered. Skipping."
                )
                continue
            worker.add_new_algorithm(algo_class())

        if not worker.algorithms:
            print(
                f"[AnalysisEngineMaster] ROI {roi_id + 1}: no valid algorithms loaded. "
                f"Worker not started."
            )
            return

        worker.daemon = True  # Dies automatically when the main app closes
        worker.start()

        self.workers[roi_id] = {
            "process": worker,
            "input_q": input_queue,
        }

    def remove_roi_stream(self, roi_id: int):
        """Safely terminate a worker process and free the CPU core."""
        if roi_id not in self.workers:
            return

        try:
            # Send the poison pill
            self.workers[roi_id]["input_q"].put_nowait(None)
        except Exception:
            pass

        worker_proc = self.workers[roi_id]["process"]
        worker_proc.join(timeout=1.0)

        if worker_proc.is_alive():
            worker_proc.terminate()

        del self.workers[roi_id]

    def process_frame(self, frame: np.ndarray, roi_list: list):
        """
        Slice the full video frame into ROI crops and dispatch each crop
        to its corresponding worker queue (non-blocking).
        """
        for i, (x, y, w, h) in enumerate(roi_list):
            if i in self.workers:
                crop = frame[y : y + h, x : x + w]
                if crop.size > 0:
                    try:
                        self.workers[i]["input_q"].put_nowait(crop)
                    except Exception:
                        pass  # Drop frame gracefully if the 15-frame queue is full

    def shutdown_all(self):
        """Terminate all active worker processes cleanly."""
        for roi_id in list(self.workers.keys()):
            self.remove_roi_stream(roi_id)
