import json
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Slot
from collections import deque

class LogBookInterface(QWidget):
    """
    A live UI table that displays arriving log events from the GlobalDataHub.
    Allows toggling between 'All Data' and 'Important Data' modes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Analytical Log Book")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #121212; color: #dddddd;")
        
        self.current_mode = "all"  # 'all' or 'important'
        
        # --- UI Build ---
        layout = QVBoxLayout(self)
        
        # Button bar for mode switching
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(0)  # Tightly stacked next to each other
        self.btn_all = QPushButton("All Data")
        self.btn_velocity = QPushButton("Velocity Tracker")
        self.btn_important = QPushButton("Important Data")
        
        # Force toggle button styles
        self.btn_all.setCheckable(True)
        self.btn_velocity.setCheckable(True)
        self.btn_important.setCheckable(True)
        self.btn_all.setChecked(True)
        
        # Consistent styling mimicking segmented controls
        base_style = """
            QPushButton { padding: 6px 16px; border: 1px solid #444; font-weight: bold; background-color: #2b2b2b; }
            QPushButton:first-child { border-top-left-radius: 4px; border-bottom-left-radius: 4px; border-right: none; }
            QPushButton:last-child { border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-left: none; }
            QPushButton:not(:first-child):not(:last-child) { border-right: none; }
        """
        self.btn_all.setStyleSheet(base_style + "QPushButton:checked { background-color: #555555; color: white; }")
        self.btn_velocity.setStyleSheet(base_style + "QPushButton:checked { background-color: #005577; color: white; }")
        self.btn_important.setStyleSheet(base_style + "QPushButton:checked { background-color: #8B0000; color: white; }")
        
        btn_layout.addWidget(self.btn_all)
        btn_layout.addWidget(self.btn_velocity)
        btn_layout.addWidget(self.btn_important)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Real-time Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "ROI ID", "Algorithm", "Data"])
        
        # Table configurations
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().hide()
        
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1e1e1e; gridline-color: #333; }"
            "QHeaderView::section { background-color: #2b2b2b; padding: 4px; border: 1px solid #333; font-weight: bold; }"
        )
        layout.addWidget(self.table)
        
        # Memory tracking prevents out-of-memory array leaks
        self.max_memory_logs = 2000
        self.max_ui_rows = 500
        
        self.all_logs = deque(maxlen=self.max_memory_logs)
        self.velocity_logs = deque(maxlen=self.max_memory_logs)
        self.important_logs = deque(maxlen=self.max_memory_logs)
        
        # Bindings
        self.btn_all.clicked.connect(self._show_all)
        self.btn_velocity.clicked.connect(self._show_velocity)
        self.btn_important.clicked.connect(self._show_important)

    def _show_all(self):
        if self.current_mode == "all": return
        self.current_mode = "all"
        self._update_buttons()
        self._refresh_table()

    def _show_velocity(self):
        if self.current_mode == "velocity": return
        self.current_mode = "velocity"
        self._update_buttons()
        self._refresh_table()
        
    def _show_important(self):
        if self.current_mode == "important": return
        self.current_mode = "important"
        self._update_buttons()
        self._refresh_table()

    def _update_buttons(self):
        self.btn_all.setChecked(self.current_mode == "all")
        self.btn_velocity.setChecked(self.current_mode == "velocity")
        self.btn_important.setChecked(self.current_mode == "important")

    def _refresh_table(self):
        self.table.setRowCount(0)
        
        if self.current_mode == "all":
            source = self.all_logs
        elif self.current_mode == "velocity":
            source = self.velocity_logs
        else:
            source = self.important_logs
        
        # Prevent blocking if millions of updates exist by tracking capacity
        # (could introduce table capacity limits natively later if required)
        for entry in list(source)[-200:]: # Draw at most last 200 items smoothly on change
            self._add_row_to_table(entry)
            
    def _add_row_to_table(self, entry: dict):
        row = self.table.rowCount()
        
        # Prevent UI DOM memory bloat by strict purging of oldest visual DOM rows
        if row >= self.max_ui_rows:
            self.table.removeRow(0)
            row = self.max_ui_rows - 1

        self.table.insertRow(row)
        
        # Format Timestamp (Drop Date, Round 2 decimals on seconds)
        try:
            dt = datetime.fromisoformat(entry["timestamp"])
            time_str = dt.strftime("%H:%M:%S.%f")[:-4]
        except Exception:
            time_str = str(entry.get("timestamp", ""))
            
        roi_str = f"ROI {entry.get('roi_id', 0)}"
        algo_str = str(entry.get("algorithm", ""))
        
        # Strip large numbers, format data neatly into JSON text dict
        formatted_data = {k: round(v, 4) if isinstance(v, float) else v for k,v in entry.get("data", {}).items() if "hist" not in k}
        data_str = json.dumps(formatted_data)
        
        t_time = QTableWidgetItem(time_str)
        t_roi = QTableWidgetItem(roi_str)
        t_algo = QTableWidgetItem(algo_str)
        t_data = QTableWidgetItem(data_str)
        
        # Optional: Color styling
        level = entry.get("level", 1)
        alert_color = None
        
        if level >= 5:
            alert_color = QColor(100, 30, 30)  # Red for anomalies
        elif level == 2:
            alert_color = QColor(20, 60, 80)   # Blue for velocity drops
            
        if alert_color:
            t_time.setBackground(alert_color)
            t_roi.setBackground(alert_color)
            t_algo.setBackground(alert_color)
            t_data.setBackground(alert_color)
            
        self.table.setItem(row, 0, t_time)
        self.table.setItem(row, 1, t_roi)
        self.table.setItem(row, 2, t_algo)
        self.table.setItem(row, 3, t_data)
        
        # Stick to bottom if we are at bottom
        self.table.scrollToBottom()

    @Slot(dict)
    def push_log(self, entry: dict):
        """Slot designed to ingest the live broadcast data object directly"""
        self.all_logs.append(entry)
        level = entry.get("level", 1)
        
        is_important = level >= 5
        is_velocity  = level == 2
        
        if is_important:
            self.important_logs.append(entry)
        if is_velocity:
            self.velocity_logs.append(entry)
            
        # UI rendering update check
        if self.current_mode == "all":
            self._add_row_to_table(entry)
        elif self.current_mode == "important" and is_important:
            self._add_row_to_table(entry)
        elif self.current_mode == "velocity" and is_velocity:
            self._add_row_to_table(entry)
