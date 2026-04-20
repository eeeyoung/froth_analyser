"""
log_book.py — Real-time Logging system for analysis data.

Stores processed pipeline results into persistent real-time logs.
Supports priority levelling to fork anomalies into dedicated IMPORTANT logs.
"""

import os
import json
import time
from datetime import datetime
from PySide6.QtCore import QObject, Signal

class LogLevel:
    INFO = 1       # Standard raw data
    VELOCITY = 2   # LK Algorithm structural analytics
    IMPORTANT = 5  # Anomalies, rapid changes, critical events

class LogBook(QObject):
    """
    Handles file-based storage of chronological data chunks.
    All data goes into the 'raw' log. Data hitting Level 2+ is also mirrored
    into the 'important' log.
    """
    
    log_ready = Signal(dict)
    
    def __init__(self, log_dir="logs", parent=None):
        super().__init__(parent)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create session-specific log files using timestamp
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_log_path = os.path.join(self.log_dir, f"session_{session_time}_raw.jsonl")
        self.important_log_path = os.path.join(self.log_dir, f"session_{session_time}_IMPORTANT.jsonl")
        
    def record(self, level: int, roi_id: int, algorithm: str, data_payload: dict):
        """
        Record a piece of data to the required logs based on its priority level.
        """
        log_entry = {
            "system_time": time.time(),
            "timestamp": datetime.now().isoformat(),
            "roi_id": roi_id + 1,
            "algorithm": algorithm,
            "level": level,
            "data": data_payload
        }
        
        entry_json = json.dumps(log_entry) + "\n"
        
        # 1. Always append to basic raw log
        with open(self.raw_log_path, "a") as raw_file:
            raw_file.write(entry_json)
            
        # 2. Mirror to important log if level qualifies
        if level >= LogLevel.IMPORTANT:
            with open(self.important_log_path, "a") as imp_file:
                imp_file.write(entry_json)
                print("IMPORTANT LOG:", entry_json)

        self._release_to_logbook(log_entry, algorithm, data_payload)

    def _release_to_logbook(self, log_entry: dict, algorithm: str, data_payload: dict):
        # 3. Create a filtered copy for real-time UI components
        ui_entry = log_entry.copy()
        filtered_data = {}
    
        if algorithm == "LBPAlgorithm":
            filtered_data = {
                "pc1": data_payload.get("pc1"),
                "pc2": data_payload.get("pc2"),
                "t_squared": data_payload.get("t_squared")
            }
        elif algorithm == "LucasKanadeAlgorithm":
            filtered_data = {
                "velocity": data_payload.get("velocity"),
                "velocity_ready": data_payload.get("velocity_ready")
            }
        else:
            filtered_data = data_payload.copy()
            
        ui_entry["data"] = filtered_data
        
        self.log_ready.emit(ui_entry)
