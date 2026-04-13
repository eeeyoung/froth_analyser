import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from queue import Empty

class ROILucasKanadeWorker(Process):
    """
    Independent background process dedicated to tracking a single ROI stream.
    Runs identically parallel on a different CPU core.
    Implements Lucas-Kanade Optical Flow to compute average pixel velocity.
    """
    def __init__(self, roi_id, input_queue, output_queue):
        super().__init__()
        self.roi_id = roi_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        
    def run(self):
        """The main execution loop for this core."""
        # Config for Lucas-Kanade math
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Config for finding the sharp bubble corners to track
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7
        )

        old_gray = None
        p0 = None
        
        print(f"[Worker {self.roi_id+1}] Multi-core processor booted up successfully.")
        
        while True:
            try:
                # 1. Grab crop from pipe (Timeout allows listening for termination signals)
                frame = self.input_queue.get(timeout=1.0)
                
                # Poison Pill routing to die cleanly
                if frame is None: 
                    break
                    
                # 2. Conversion to grayscale for tracking
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 3. Harris Corners detection (Find new bubbles if starting or lost them)
                if old_gray is None or p0 is None or len(p0) < 15:
                    old_gray = frame_gray
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    if p0 is None:
                        continue # Skip math until bubbles enter the ROI
                        
                # 4. Calculate Optical Flow against the previous frame
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # 5. Math extraction
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        # Vector calculation
                        displacements = good_new - good_old
                        avg_dx, avg_dy = np.mean(displacements, axis=0) # Average movement
                        
                        # Package RAW PIXELS for the global Data Hub
                        result = {
                            "roi_id": self.roi_id,
                            "timestamp": time.time(),
                            "dx_pixels": float(avg_dx),
                            "dy_pixels": float(avg_dy),
                            "features_tracked": len(good_new)
                        }
                        
                        # Fire into IPC queue to the Data Hub without waiting
                        try:
                            self.output_queue.put_nowait(result)
                        except:
                            pass
                            
                    # Update local state for the next incoming frame
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                    
            except Empty:
                continue # Normal timeout, loop and keep waiting
            except Exception as e:
                print(f"[Worker {self.roi_id+1}] Process Crashed: {str(e)}")
                break
                
        print(f"[Worker {self.roi_id+1}] Shutting down safely and releasing memory.")


class AnalysisEngineMaster:
    """
    Manager running on the Main Thread. 
    It calculates absolutely nothing; it just hires/fires workers and routes video chunks.
    """
    def __init__(self, data_hub):
        self.workers = {} # Maps roi_id -> dict of Process and Input Ques
        self.data_hub = data_hub 

    def add_roi_stream(self, roi_id):
        """Spawns a new parallel worker process for a newly drawn ROI."""
        if roi_id in self.workers:
            return 
            
        # Maxsize prevents old frames from backing up in memory if the worker process is slow!
        # This acts as our "Skip Frame" throttle logic.
        input_queue = Queue(maxsize=15) 
        
        # Connect directly to the Data Hub's global collection bin
        worker = ROILucasKanadeWorker(roi_id, input_queue, self.data_hub.collection_queue)
        worker.daemon = True # Dies if main app closes
        worker.start()
        
        self.workers[roi_id] = {
            "process": worker,
            "input_q": input_queue
        }

    def remove_roi_stream(self, roi_id):
        """Safely murders a worker process to free up the CPU core."""
        if roi_id in self.workers:
            try:
                # 1. Send the termination signal ("None" object)
                self.workers[roi_id]["input_q"].put_nowait(None)
            except:
                pass
                
            # 2. Wait up to 1 second for it to die nicely
            worker_proc = self.workers[roi_id]["process"]
            worker_proc.join(timeout=1.0)
            
            # 3. If it's frozen, force kill it
            if worker_proc.is_alive():
                worker_proc.terminate()
                
            del self.workers[roi_id]

    def process_frame(self, frame: np.ndarray, roi_list: list):
        """Slices the main video frame and throws the pieces down the Queues."""
        for i, (x, y, w, h) in enumerate(roi_list):
            if i in self.workers:
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0:
                    try:
                        self.workers[i]["input_q"].put_nowait(crop)
                    except:
                        pass # Drop frame gracefully if worker's 15-frame queue is full
                        
    def shutdown_all(self):
        """Closes all parallel pipes securely."""
        for roi_id in list(self.workers.keys()):
            self.remove_roi_stream(roi_id)
