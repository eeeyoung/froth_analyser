"""
pca_handler.py — Logic for managing LBP texture baseline collection and PCA projection.
"""

import time
import numpy as np
from sklearn.decomposition import PCA

class LBPPCAHandler:
    """
    Handles the state transition for an ROI from 'baseline' (collecting histograms)
    to 'monitoring' (projecting onto PCA components).
    """
    def __init__(self, baseline_duration: float = 1.5):
        self.baseline_duration = baseline_duration
        self.status = "baseline"
        self.start_time = time.time()
        self.data = []
        self.pca = None
        self.mean_pc = None
        self.std_pc = None

    def process_frame(self, combined_hist: np.ndarray, roi_id: int) -> dict:
        """
        Process a single frame of LBP histogram data.
        Returns a dict with PCA projections or baseline status.
        """
        elapsed = time.time() - self.start_time

        if self.status == "baseline":
            self.data.append(combined_hist)
            
            if elapsed >= self.baseline_duration:
                X = np.array(self.data)
                if len(X) >= 2:
                    self.pca = PCA(n_components=2)
                    self.pca.fit(X)
                    
                    # Compute confidence interval base measures
                    X_pca = self.pca.transform(X)
                    self.mean_pc = np.mean(X_pca, axis=0)
                    self.std_pc = np.std(X_pca, axis=0)
                    
                    self.status = "monitoring"
                    print(f"\n--- [LBP PCA] ROI {roi_id + 1} Baseline Complete! Used {len(X)} frames over {elapsed:.2f} seconds. ---\n")
                else:
                    # Not enough data, maybe keep trying or handle error
                    pass

            return {
                "is_baseline": True,
                "pc1": None,
                "pc2": None,
                "elapsed": elapsed
            }
        else:
            # Monitoring phase
            coords = self.pca.transform(combined_hist.reshape(1, -1))[0]
            
            # Calculate Hotelling's T-squared distance (using PCA axes since they are orthogonal)
            dist_x = ((coords[0] - self.mean_pc[0]) / (self.std_pc[0] + 1e-9)) ** 2
            dist_y = ((coords[1] - self.mean_pc[1]) / (self.std_pc[1] + 1e-9)) ** 2
            t_squared = float(dist_x + dist_y)
            
            # Calculate Q-statistic (SPE)
            reconstructed = self.pca.inverse_transform(coords.reshape(1, -1))[0]
            q_statistic = float(np.sum((combined_hist - reconstructed) ** 2))
            
            # 95% confidence chi-square threshold for 2 degrees of freedom is approx 5.991
            is_anomaly = t_squared > 5.991
            
            return {
                "is_baseline": False,
                "pc1": float(coords[0]),
                "pc2": float(coords[1]),
                "mean_pc1": float(self.mean_pc[0]),
                "mean_pc2": float(self.mean_pc[1]),
                "std_pc1": float(self.std_pc[0]),
                "std_pc2": float(self.std_pc[1]),
                "var_pc1": float(self.pca.explained_variance_ratio_[0]),
                "var_pc2": float(self.pca.explained_variance_ratio_[1]),
                "t_squared": t_squared,
                "q_statistic": q_statistic,
                "is_anomaly": is_anomaly,
                "elapsed": elapsed
            }
