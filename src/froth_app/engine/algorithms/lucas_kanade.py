"""
lucas_kanade.py — Lucas-Kanade Sparse Optical Flow analysis algorithm.

Implements BaseAnalysisAlgorithm to track the average pixel velocity of
bubble features inside a single ROI using the pyramidal Lucas-Kanade method.

Algorithm summary
-----------------
1. Convert incoming BGR frame to grayscale.
2. If no tracked points exist (or too few remain), detect new Shi-Tomasi
   corner features as proxy tracking targets for bubble edges.
3. Run cv2.calcOpticalFlowPyrLK() to estimate where those corners moved.
4. Compute the mean (dx, dy) displacement vector across all good matches.
5. Return the raw pixel displacement; calibration to real-world units is
   handled downstream by GlobalDataHub.
"""

import cv2
import numpy as np
from froth_app.engine.algorithms.base import BaseAnalysisAlgorithm


class LucasKanadeAlgorithm(BaseAnalysisAlgorithm):
    """
    Sparse Lucas-Kanade Optical Flow velocity estimator.

    Tracks Shi-Tomasi corners across consecutive frames and reports the
    average displacement vector in pixels per frame for a single ROI crop.
    """

    # Minimum number of tracked points before re-detecting features
    _MIN_FEATURES = 15

    def __init__(self):
        # Lucas-Kanade solver configuration
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        # Shi-Tomasi (goodFeaturesToTrack) configuration
        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7,
        )
        # Internal state: previous frame and tracked points
        self._old_gray: np.ndarray | None = None
        self._p0: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseAnalysisAlgorithm interface
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """
        Process one BGR crop and return a velocity result dict, or None
        when there is insufficient data to compute a displacement.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Feature detection phase ---
        # Trigger on first frame, or when tracked points drop too low.
        no_prev = self._old_gray is None
        too_few = self._p0 is None or len(self._p0) < self._MIN_FEATURES
        if no_prev or too_few:
            self._old_gray = frame_gray
            self._p0 = cv2.goodFeaturesToTrack(
                frame_gray, mask=None, **self._feature_params
            )
            # Not enough data yet; skip until next frame brings a second image
            return None

        # --- Optical flow phase ---
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray, frame_gray, self._p0, None, **self._lk_params
        )

        if p1 is None or st is None:
            # Tracking completely lost; trigger re-detection on the next frame
            self.reset()
            return None

        good_new = p1[st == 1]
        good_old = self._p0[st == 1]

        if len(good_new) == 0:
            self.reset()
            return None

        # --- Displacement math ---
        displacements = good_new - good_old
        avg_dx, avg_dy = np.mean(displacements, axis=0)

        result = {
            "dx_pixels": float(avg_dx),
            "dy_pixels": float(avg_dy),
            "features_tracked": len(good_new),
        }

        # Advance internal state for the next frame
        self._old_gray = frame_gray.copy()
        self._p0 = good_new.reshape(-1, 1, 2)

        return result

    def reset(self) -> None:
        """Clear all internal state so the next frame starts fresh."""
        self._old_gray = None
        self._p0 = None
