"""
analyzer.py — ROIWorker and AnalysisEngineMaster

Algorithm Registry
------------------
ALGORITHM_REGISTRY maps integer IDs to algorithm classes.
To register a new algorithm, add an entry here.

    1  → LucasKanadeAlgorithm
    2  → LBPAlgorithm
    (future entries go here)

ROIWorker
---------
An independent multiprocessing.Process that runs a *list* of
BaseAnalysisAlgorithm instances on a dedicated CPU core for one ROI stream.

Frame data arrives via shared memory (zero-copy): the main thread writes the
crop into a FrameSharedBuffer and puts a tiny FrameMeta namedtuple on the
input_queue. The worker reconnects to the SharedMemory block by name and
reconstructs the numpy array with a single memcpy — no pickle, no pipe data.

Multiple algorithms can be loaded before the process starts via
add_new_algorithm(); all of them run sequentially on every incoming frame
and each posts its own result to the DataHub.

AnalysisEngineMaster
--------------------
Main-thread manager. Accepts a list of algorithm IDs per ROI, resolves
them through ALGORITHM_REGISTRY, spawns an ROIWorker, and loads it up
before starting the process. Handles routing and lifecycle management.

Shared memory lifecycle
-----------------------
Each ROI gets its own FrameSharedBuffer, allocated lazily on the first frame
so the buffer is sized to the actual crop dimensions. The master is the sole
owner and calls both close() and unlink() on teardown. Workers only close().

    Example — run LK (id=1) on ROI 0, and two algorithms on ROI 1:
        master.add_roi_stream(roi_id=0, algorithm_ids=[1])
        master.add_roi_stream(roi_id=1, algorithm_ids=[1, 2])
"""

import time
import numpy as np
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty

from froth_app.engine.algorithms.base import BaseAnalysisAlgorithm
from froth_app.engine.algorithms.lucas_kanade import LucasKanadeAlgorithm
from froth_app.engine.algorithms.lbp import LBPAlgorithm
from froth_app.engine.frame_buffer import FrameSharedBuffer, FrameMeta


# ---------------------------------------------------------------------------
# Algorithm Registry
# ---------------------------------------------------------------------------
ALGORITHM_REGISTRY: dict[int, type[BaseAnalysisAlgorithm]] = {
    1: LucasKanadeAlgorithm,
    2: LBPAlgorithm,
}


class ROIWorker(Process):
    """
    Independent background process dedicated to analysing a single ROI stream.
    Runs on a separate CPU core (true parallelism via multiprocessing).

    Receives FrameMeta signals via input_queue (tiny, ~40 bytes each), then
    reads the actual pixel data directly from the named SharedMemory block —
    no pickle serialization of frame arrays.

    IMPORTANT: add_new_algorithm() must be called BEFORE start(). After the
    process is spawned, parent-side mutations are not visible to the child.
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
        Must be called before start() — multiprocessing copies state at spawn time.
        """
        self.algorithms.append(algorithm)

    def run(self):
        """Main execution loop — reads frames from shared memory, runs algorithms."""
        algo_names = [type(a).__name__ for a in self.algorithms]
        print(
            f"[Worker {self.roi_id + 1} | {', '.join(algo_names) or 'no algorithms'}]"
            f" Process booted (shared memory mode)."
        )

        # SharedMemory handle — attached lazily on receipt of the first FrameMeta
        shm: SharedMemory | None = None
        current_shm_name: str | None = None

        while True:
            try:
                # 1. Grab the next FrameMeta signal (tiny — metadata only, no pixels)
                meta = self.input_queue.get(timeout=1.0)

                # Poison pill — shut down cleanly
                if meta is None:
                    break

                # 2. Attach to the shared memory block (once, or if name changes)
                if meta.name != current_shm_name:
                    if shm is not None:
                        shm.close()
                    shm = SharedMemory(name=meta.name, create=False)
                    current_shm_name = meta.name

                # 3. Read the frame crop — one memcpy, no pickle
                frame = FrameSharedBuffer.read(shm, meta)

                # 4. Run every loaded algorithm on the same frame
                for algorithm in self.algorithms:
                    result = algorithm.process_frame(frame)

                    # 5. Annotate and forward each result to the DataHub
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

        # Close handle only — master is the owner and calls unlink()
        if shm is not None:
            shm.close()
        print(f"[Worker {self.roi_id + 1}] Shutting down safely.")


class AnalysisEngineMaster:
    """
    Manager running on the main thread.
    Spawns/terminates ROIWorker processes and routes video frame crops
    to them via shared memory. Performs no computation itself.

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
        # Maps roi_id → {"process": ROIWorker, "input_q": Queue,
        #                 "buffer": FrameSharedBuffer | None}
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

        # Bounded queue — carries FrameMeta only (~40 bytes each); drop throttle
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
            "buffer": None,  # FrameSharedBuffer — allocated lazily on first frame
        }

    def remove_roi_stream(self, roi_id: int):
        """Safely terminate a worker, release its shared memory, and free the core."""
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

        # Release and destroy the shared memory block (master is owner)
        buf: FrameSharedBuffer | None = self.workers[roi_id].get("buffer")
        if buf is not None:
            buf.close()
            buf.unlink()

        del self.workers[roi_id]

    def process_frame(self, frame: np.ndarray, roi_list: list):
        """
        Slice the full video frame into ROI crops, write each into its shared
        memory buffer, and dispatch a lightweight FrameMeta signal to the
        worker queue (non-blocking). No frame pixel data goes through the queue.
        """
        for i, (x, y, w, h) in enumerate(roi_list):
            if i not in self.workers:
                continue

            crop = frame[y: y + h, x: x + w]
            if crop.size == 0:
                continue

            worker_data = self.workers[i]

            # Lazy init: allocate the shared buffer on the first frame for this ROI,
            # sized exactly to the actual crop dimensions (not a fixed maximum).
            if worker_data["buffer"] is None:
                ch = crop.shape[2] if crop.ndim == 3 else 1
                worker_data["buffer"] = FrameSharedBuffer(
                    crop.shape[0], crop.shape[1], ch, crop.dtype
                )
                print(
                    f"[AnalysisEngineMaster] ROI {i + 1}: shared buffer allocated "
                    f"({crop.shape[0]}x{crop.shape[1]}x{ch} {crop.dtype}, "
                    f"name='{worker_data['buffer'].name}')."
                )

            try:
                meta = worker_data["buffer"].write(crop)
                worker_data["input_q"].put_nowait(meta)
            except Exception:
                pass  # Drop frame gracefully if the queue is full

    def shutdown_all(self):
        """Terminate all active worker processes and free all shared memory."""
        for roi_id in list(self.workers.keys()):
            self.remove_roi_stream(roi_id)
