"""
frame_buffer.py — Zero-copy frame transport via shared memory.

FrameSharedBuffer owns one SharedMemory block sized for two slots (double
buffer). The main thread writes crop data into the inactive slot and sends
a tiny FrameMeta namedtuple through the Queue. The ROIWorker reconnects to
the same block by name and reads only from the slot indicated in the meta.

Layout
------
  [ slot_0 (slot_bytes) | slot_1 (slot_bytes) ]
  Where slot_bytes = H * W * channels * dtype_itemsize for the first crop seen.

Double-buffer guarantee
-----------------------
The writer always writes to the slot that was NOT last used by the reader.
The Queue (maxsize=15) bounds how far ahead the writer can get, ensuring the
reader always sees a consistent frame without needing a mutex.
"""

from __future__ import annotations

import numpy as np
from multiprocessing.shared_memory import SharedMemory
from collections import namedtuple


# Metadata sent through the Queue — deliberately tiny (~40 bytes when pickled)
FrameMeta = namedtuple(
    "FrameMeta",
    ["name", "slot", "h", "w", "channels", "dtype", "frame_id", "slot_bytes"],
)


class FrameSharedBuffer:
    """
    Manages a double-slot SharedMemory block for zero-copy numpy frame transport.

    Lifecycle
    ---------
    - Created by AnalysisEngineMaster (the owning process).
    - Workers attach by name: SharedMemory(name=meta.name, create=False).
    - On teardown: master calls close() then unlink(); workers call close() only.
    """

    def __init__(self, h: int, w: int, channels: int = 3, dtype=np.uint8):
        self._h = h
        self._w = w
        self._channels = channels
        self._dtype = np.dtype(dtype)
        self._slot_bytes = int(h * w * channels * self._dtype.itemsize)
        self._shm = SharedMemory(create=True, size=2 * self._slot_bytes)
        self._write_slot = 0   # Next slot to write into
        self._frame_id = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, crop: np.ndarray) -> FrameMeta:
        """
        Copy *crop* into the inactive slot and return a FrameMeta for the Queue.

        The crop is clipped to (h, w) if it arrives slightly smaller due to
        frame-edge ROIs. Dtype is cast to match the buffer if needed.
        """
        h = min(crop.shape[0], self._h)
        w = min(crop.shape[1], self._w)
        channels = crop.shape[2] if crop.ndim == 3 else 1

        # Select the slot the reader is NOT currently using
        write_slot = 1 - self._write_slot
        offset = write_slot * self._slot_bytes

        # View the slot as a writable numpy array backed by shared memory
        dest = np.ndarray(
            (self._h, self._w, self._channels),
            dtype=self._dtype,
            buffer=self._shm.buf,
            offset=offset,
        )

        # Fast C-level memcpy via numpy — no pickle, no pipe data
        src = crop[:h, :w]
        if src.dtype != self._dtype:
            src = src.astype(self._dtype)
        dest[:h, :w] = src

        # Commit: flip the active slot and bump the monotonic counter
        self._write_slot = write_slot
        self._frame_id += 1

        return FrameMeta(
            name=self._shm.name,
            slot=write_slot,
            h=h,
            w=w,
            channels=channels,
            dtype=str(self._dtype),
            frame_id=self._frame_id,
            slot_bytes=self._slot_bytes,
        )

    @staticmethod
    def read(shm: SharedMemory, meta: FrameMeta) -> np.ndarray:
        """
        Read the frame described by *meta* from the shared block *shm*.

        Returns a freshly allocated numpy array (.copy()) so algorithms can
        mutate it freely and the main thread can overwrite the slot immediately.
        """
        offset = meta.slot * meta.slot_bytes
        n_elements = meta.h * meta.w * meta.channels
        raw = np.frombuffer(
            shm.buf, dtype=np.dtype(meta.dtype), count=n_elements, offset=offset
        )
        return raw.reshape(meta.h, meta.w, meta.channels).copy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release the handle — call from every process that attached."""
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        """Destroy the OS-level block — call ONLY from the owning process."""
        try:
            self._shm.unlink()
        except Exception:
            pass
