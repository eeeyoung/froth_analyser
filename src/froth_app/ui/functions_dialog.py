"""
functions_dialog.py
Dialog for toggling analysis algorithms per ROI.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QWidget,
    QCheckBox, QDialogButtonBox
)
from src.froth_app.core.algorithm_state import AlgorithmStateManager
from src.froth_app.core.data_hub import GlobalDataHub

_ALGO_LBP = 2

class FunctionsDialog(QDialog):
    """
    Popup listing available analysis functions.
    Checkbox states are pre-populated from AlgorithmStateManager and written
    back to it when the user confirms.
    """
    def __init__(self, algo_state: AlgorithmStateManager, data_hub: GlobalDataHub, parent=None):
        super().__init__(parent)
        self._algo_state = algo_state
        self._data_hub = data_hub

        self.setWindowTitle("Analysis Functions")
        self.setFixedSize(320, 220)
        self.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0;")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        title = QLabel("Select functions to run on each ROI:")
        title.setStyleSheet("font-size: 11px; color: #aaaaaa;")
        layout.addWidget(title)

        # Build spinbox for LBP Baseline config
        self._lbp_duration_spinbox = QDoubleSpinBox()
        self._lbp_duration_spinbox.setRange(0.5, 100.0)
        self._lbp_duration_spinbox.setSingleStep(0.5)
        self._lbp_duration_spinbox.setValue(self._data_hub.baseline_duration)
        self._lbp_duration_spinbox.setSuffix(" s")
        self._lbp_duration_spinbox.setStyleSheet(
            "QDoubleSpinBox { font-size: 12px; background-color: #2c2c2c; "
            "color: white; border: 1px solid #444; border-radius: 3px; padding: 2px; }"
        )
        self._lbp_duration_container = QWidget()
        dur_layout = QHBoxLayout(self._lbp_duration_container)
        dur_layout.setContentsMargins(24, 0, 0, 0)
        lbl = QLabel("↳ Baseline duration:")
        lbl.setStyleSheet("font-size: 11px; color: #888888;")
        dur_layout.addWidget(lbl)
        dur_layout.addWidget(self._lbp_duration_spinbox)
        dur_layout.addStretch()

        # Build one checkbox per algorithm, pre-populated from state
        self._checkboxes: dict[int, QCheckBox] = {}
        for algo_id, label in AlgorithmStateManager.ALGORITHM_LABELS.items():
            chk = QCheckBox(label)
            chk.setChecked(algo_state.is_active(algo_id))
            chk.setStyleSheet("font-size: 12px;")
            layout.addWidget(chk)
            self._checkboxes[algo_id] = chk
            
            if algo_id == _ALGO_LBP:
                layout.addWidget(self._lbp_duration_container)
                self._lbp_duration_container.setVisible(chk.isChecked())
                chk.toggled.connect(self._lbp_duration_container.setVisible)

        # Confirm button
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: white; "
            "border-radius: 4px; padding: 4px 14px; }"
            "QPushButton:hover { background-color: #555555; }"
        )
        btn_box.accepted.connect(self._on_confirm)
        layout.addWidget(btn_box)

    def _on_confirm(self):
        """Write checkbox states back to AlgorithmStateManager, then close."""
        snapshot = {
            algo_id: chk.isChecked()
            for algo_id, chk in self._checkboxes.items()
        }
        self._algo_state.apply_snapshot(snapshot)
        
        # Sync the baseline duration value to the data hub seamlessly
        self._data_hub.update_baseline_duration(self._lbp_duration_spinbox.value())
        
        self.accept()
