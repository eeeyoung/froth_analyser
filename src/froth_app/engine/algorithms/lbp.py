"""
lbp.py — Local Binary Pattern RGB (LBP-RGB) texture analysis algorithm.

Implements BaseAnalysisAlgorithm to characterise the surface texture of
froth/bubbles inside a single ROI using LBP applied independently to each
RGB colour channel.

Algorithm summary
-----------------
1. Split the incoming BGR frame into B, G, R planes.
2. Compute a uniform LBP map on each plane independently using a circular
   neighbourhood (P neighbours on radius R).
3. Build a normalised 256-bin histogram for each channel.
4. Compute the chi-square distance between the current combined histogram
   and the one from the previous frame as a scalar "texture change score".
5. Return the three histograms and the change score.

Output dict keys
----------------
    lbp_r_hist            np.ndarray(256,) — normalised R-channel histogram
    lbp_g_hist            np.ndarray(256,) — normalised G-channel histogram
    lbp_b_hist            np.ndarray(256,) — normalised B-channel histogram
    texture_change_score  float            — chi-sq distance vs. previous frame
                                             (0.0 on the very first frame)
    features_tracked      int              — ROI pixel count (DataHub compat key)

Visualisation notes
-------------------
- Plot lbp_r/g/b_hist as 3 overlapping line charts (R/G/B) to inspect the
  texture distribution of the froth surface each frame.
- Plot texture_change_score as a rolling time series; spikes indicate sudden
  froth phase transitions (e.g. wet → dry froth).
"""

import cv2
import numpy as np
from froth_app.engine.algorithms.base import BaseAnalysisAlgorithm


def _compute_lbp(gray: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    Compute a circular LBP map for a single-channel (grayscale) image.

    Uses bilinear interpolation for non-integer neighbour coordinates and
    produces codes in the range [0, 2^P − 1].

    Parameters
    ----------
    gray : np.ndarray (H, W), uint8
        Single-channel image.
    P : int
        Number of neighbours on the sampling circle (default 8).
    R : int
        Radius of the sampling circle in pixels (default 1).

    Returns
    -------
    np.ndarray (H, W), uint8
        LBP-encoded image.
    """
    H, W = gray.shape
    lbp = np.zeros((H, W), dtype=np.uint8)
    gray_f = gray.astype(np.float32)

    # Base coordinate grids — shape (H, W) float32, required by cv2.remap
    col_coords = np.arange(W, dtype=np.float32)   # (W,)
    row_coords = np.arange(H, dtype=np.float32)   # (H,)
    base_x, base_y = np.meshgrid(col_coords, row_coords)  # both (H, W)

    for p in range(P):
        angle = 2 * np.pi * p / P
        map_x = (base_x + R * np.cos(angle)).astype(np.float32)
        map_y = (base_y - R * np.sin(angle)).astype(np.float32)

        neighbour = cv2.remap(
            gray_f, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Threshold: neighbour >= centre → bit p = 1
        lbp |= ((neighbour >= gray_f).astype(np.uint8) << p)

    return lbp


def _histogram(lbp_map: np.ndarray, bins: int = 256) -> np.ndarray:
    """Return a normalised histogram of LBP codes."""
    hist, _ = np.histogram(lbp_map.ravel(), bins=bins, range=(0, bins))
    total = hist.sum()
    return hist.astype(np.float32) / total if total > 0 else hist.astype(np.float32)


def _chi_square(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-square distance between two normalised histograms."""
    denom = h1 + h2
    mask = denom > 0
    return float(0.5 * np.sum(((h1[mask] - h2[mask]) ** 2) / denom[mask]))


class LBPAlgorithm(BaseAnalysisAlgorithm):
    """
    Local Binary Pattern RGB texture descriptor.

    Characterises froth surface texture per frame. Particularly useful for
    detecting froth phase changes and surface quality variations that optical
    flow alone cannot capture.
    """

    def __init__(self, P: int = 8, R: int = 1):
        """
        Parameters
        ----------
        P : int
            Number of LBP neighbour points on the sampling circle (default 8).
        R : int
            Radius of the sampling circle in pixels (default 1).
        """
        self._P = P
        self._R = R
        # Combined histogram from the previous frame (R+G+B concatenated)
        self._prev_hist: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseAnalysisAlgorithm interface
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """
        Compute LBP-RGB descriptors for one BGR crop.

        Returns a result dict on every frame (including the first, where
        texture_change_score is 0.0 because there is no previous reference).
        """
        if frame is None or frame.size == 0:
            return None

        # --- Split channels (OpenCV gives BGR order) ---
        b_plane, g_plane, r_plane = cv2.split(frame)

        # --- Per-channel LBP maps ---
        lbp_r = _compute_lbp(r_plane, self._P, self._R)
        lbp_g = _compute_lbp(g_plane, self._P, self._R)
        lbp_b = _compute_lbp(b_plane, self._P, self._R)

        # --- Normalised histograms ---
        hist_r = _histogram(lbp_r)
        hist_g = _histogram(lbp_g)
        hist_b = _histogram(lbp_b)

        # --- Texture change score vs. previous frame ---
        combined = np.concatenate([hist_r, hist_g, hist_b])
        if self._prev_hist is None:
            score = 0.0
        else:
            score = _chi_square(combined, self._prev_hist)
        self._prev_hist = combined

        return {
            "lbp_r_hist":           hist_r,
            "lbp_g_hist":           hist_g,
            "lbp_b_hist":           hist_b,
            "texture_change_score": score,
            "features_tracked":     int(frame.shape[0] * frame.shape[1]),
        }

    def reset(self) -> None:
        """Clear previous-frame histogram so the next frame starts fresh."""
        self._prev_hist = None
