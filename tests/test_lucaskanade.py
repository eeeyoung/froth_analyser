import sys
sys.path.insert(0, "src")
import numpy as np
from froth_app.engine.algorithms.lucas_kanade import LucasKanadeAlgorithm

algo = LucasKanadeAlgorithm()
dummy = np.zeros((100, 100, 3), dtype=np.uint8)

# Frame 1: no previous — expect None
result = algo.process_frame(dummy)
print(f"Frame 1 (no prev): {result}")
assert result is None, "Expected None on first frame"

# Frame 2: has a previous, but blank frame has no features — expect None
result = algo.process_frame(dummy)
print(f"Frame 2 (blank):   {result}")

# Frames with actual content
real_frame = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
result = algo.process_frame(real_frame)
print(f"Frame 3 (random):  {result}")

result2 = algo.process_frame(real_frame)
print(f"Frame 4 (random):  {result2}")

# Test reset()
algo.reset()
result_after_reset = algo.process_frame(real_frame)
print(f"After reset():     {result_after_reset}")
assert result_after_reset is None, "Expected None right after reset()"

print("\n=== LUCAS KANADE TEST PASSED ===")
