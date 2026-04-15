"""
algorithm_state.py — AlgorithmStateManager

Central, single-source-of-truth for which analysis algorithms are currently
activated by the user. The GUI reads from this manager to pre-populate the
FunctionsDialog checkboxes, and writes back to it on Confirm.
The analyzer reads from it when spawning a new ROI worker.

Algorithm IDs must mirror ALGORITHM_REGISTRY in analyzer.py:
    1  →  LucasKanadeAlgorithm
    2  →  LBPAlgorithm
"""


class AlgorithmStateManager:
    """
    Lightweight state container — no Qt dependency, no threading.

    All algorithms are activated by default on construction.
    """

    # Human-readable labels keyed by the same IDs as ALGORITHM_REGISTRY.
    ALGORITHM_LABELS: dict[int, str] = {
        1: "Movement detection (LucasKanade)",
        2: "LBP-RGB feature extraction",
    }

    def __init__(self):
        # Start with every algorithm enabled (True)
        self._active: dict[int, bool] = {
            algo_id: True for algo_id in self.ALGORITHM_LABELS
        }

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def is_active(self, algo_id: int) -> bool:
        """Return True if the given algorithm is currently enabled."""
        return self._active.get(algo_id, False)

    def active_ids(self) -> list[int]:
        """Return a sorted list of currently enabled algorithm IDs."""
        return sorted(k for k, v in self._active.items() if v)

    def snapshot(self) -> dict[int, bool]:
        """Return a copy of the full {id: bool} state dict."""
        return dict(self._active)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def set_active(self, algo_id: int, active: bool) -> None:
        """Enable or disable one algorithm by ID."""
        if algo_id in self._active:
            self._active[algo_id] = active

    def apply_snapshot(self, state: dict[int, bool]) -> None:
        """
        Bulk-update from a {id: bool} dict (e.g. from FunctionsDialog).
        Unknown IDs are silently ignored.
        """
        for algo_id, active in state.items():
            self.set_active(algo_id, active)
