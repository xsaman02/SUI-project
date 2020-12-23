import numpy as np
import time

dataset = np.random.random((1000, 4))
timesets = []

for x in range(1000000):
	dp = np.random.random((4))
	start = time.perf_counter()
	_ = np.linalg.norm(dataset - dp, axis=0)
	timesets.append(time.perf_counter() - start)

print(np.asarray(timesets).mean())
