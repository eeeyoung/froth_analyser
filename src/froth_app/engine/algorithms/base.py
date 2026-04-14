"""
base.py — Abstract base class for all RT-FFAT analysis algorithms.

Every algorithm module must subclass BaseAnalysisAlgorithm and implement
the two abstract methods below so it can be dropped into ROIWorker without
any changes to the worker or the AnalysisEngineMaster.

Output dict contract
--------------------
process_frame() must return either None (not enough data yet) or a dict
containing AT LEAST the following keys so GlobalDataHub stays compatible:

    {
        "dx_pixels":       float,  # average horizontal displacement (pixels)
        "dy_pixels":       float,  # average vertical displacement  (pixels)
        "features_tracked": int,   # number of elements tracked/detected
    }

roi_id and timestamp are injected by ROIWorker, so algorithms must NOT
include them in their returned dict.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAnalysisAlgorithm(ABC):
    """
    Abstract contract for a single-ROI frame analysis algorithm.
    Subclass this and implement process_frame() and reset() to create
    a new pluggable analysis module.
    """

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict | None:
        """
        Accept one cropped BGR frame and return a result dict or None.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image crop corresponding to a single ROI region.

        Returns
        -------
        dict
            Result dict following the output contract above.
        None
            When the algorithm cannot yet produce a result (e.g. waiting
            for a second frame, no features detected, etc.).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state.
        Called when an ROI is re-drawn or the video source is changed,
        so the algorithm starts fresh without stale state from the previous run.
        """
        ...
