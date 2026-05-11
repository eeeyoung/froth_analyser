# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RT-FFAT — Real-Time Froth Flotation Analysis Tool. A PySide6 desktop application that analyses flotation froth from live camera feeds or video files using computer vision (optical flow, LBP texture analysis, PCA anomaly detection).

## Commands

```bash
# Install dependencies (Poetry, Python 3.11)
poetry install

# Run the full application
poetry run python simple_test_gui.py

# Run the minimal placeholder window
poetry run python main.py

# Run the video source module in isolation (test widget)
poetry run python src/froth_app/core/video_source.py

# Run the Lucas-Kanade algorithm test
poetry run python tests/test_lucaskanade.py
```

## Architecture

**Multiprocessing pipeline with zero-copy shared memory.** The main thread owns the UI and an `AnalysisEngineMaster`. Each ROI gets its own `ROIWorker` process (true parallelism via `multiprocessing.Process`) running one or more algorithms. Frames flow from main thread → workers via `FrameSharedBuffer` (double-buffered `SharedMemory`, no pickle serialization of pixel data), with tiny `FrameMeta` namedtuples sent through bounded `Queue`s as signals.

**Data path:** `VideoSource` (QThread) emits BGR frames → main thread crops ROIs → `AnalysisEngineMaster.process_frame()` writes crops into per-ROI shared memory → `ROIWorker` reads via memcpy, runs algorithms, puts result dicts on output queue → `GlobalDataHub` (QThread) collects, processes (raw→clean dict), emits Qt signals, logs to disk.

### Key modules

- **`simple_test_gui.py`** — The real application window (not `main_window.py`, which is a placeholder skeleton). Wires together video source, ROI overlay, engine, plots, calibration, and log book.
- **`src/froth_app/core/`** — Pure data/state modules: `VideoSource`, `ROICoordinateManager`, `CalibrationManager`, `AlgorithmStateManager`, `LogBook`. No algorithm logic.
- **`src/froth_app/engine/`** — `AnalysisEngineMaster` (spawns/terminates workers), `ROIWorker` (per-ROI process), `FrameSharedBuffer` (zero-copy shared memory transport).
- **`src/froth_app/engine/algorithms/`** — Pluggable algorithms extending `BaseAnalysisAlgorithm`. Each implements `process_frame(crop) -> dict|None` and `reset()`. Registered in `ALGORITHM_REGISTRY` dict (1=LucasKanade, 2=LBP).
- **`src/froth_app/ui/`** — PySide6 widgets: ROI overlay (transparent drawing layer), detail popups (live crop view with motion crosshair), pyqtgraph-based plot widgets (PCA scatter, velocity timeseries, T²/Q-statistic), log book table, calibration dialogs.

### Algorithm details

- **LucasKanade (id=1):** Sparse optical flow on Shi-Tomasi corners. Reports average (dx, dy) per frame. DataHub projects displacement onto the calibrated overflow axis and accumulates 1-second velocity windows.
- **LBP (id=2):** Uniform LBP on each RGB channel independently, chi-square distance between consecutive frames. PCA on the combined 177-dim histogram (via `LBPPCAHandler`): first N seconds are baseline collection, then projects new frames onto PC1/PC2 and computes Hotelling's T² and Q-statistic (SPE) for anomaly detection. PCA model updates dynamically on a configurable interval.

### Adding a new algorithm

1. Create a class in `src/froth_app/engine/algorithms/` extending `BaseAnalysisAlgorithm`
2. Add an entry to `ALGORITHM_REGISTRY` in `analyzer.py` with a new integer ID
3. Add a label in `AlgorithmStateManager.ALGORITHM_LABELS`
4. Add `_process_<algo>()` and `_format_<algo>()` methods in `GlobalDataHub`, and register them in `_PROCESSORS`/`_FORMATTERS`
5. If needed, add a Qt Signal to `GlobalDataHub` and emit it in `_ingest()`

## Important conventions

- **Video decoding backend priority:** decord GPU (NVDEC) → decord CPU → OpenCV fallback. Live cameras always use OpenCV.
- **Shared memory lifecycle:** Only `AnalysisEngineMaster` calls `close()` + `unlink()`. Workers call `close()` only.
- **Output queue is bounded** (maxsize=60): workers silently drop results if DataHub falls behind.
- **BGR color order throughout the pipeline** — decord outputs RGB and is converted to BGR. LBP splits BGR planes from OpenCV convention.
- **LogBook writes are buffered** and flushed every 2 seconds, not per-frame. Only VELOCITY/IMPORTANT events emit UI signals.
- **Windows-specific workarounds:** `timeBeginPeriod(1)` for 1ms timer resolution, `cv2.CAP_DSHOW` for faster camera init, `freeze_support()` for multiprocessing.
