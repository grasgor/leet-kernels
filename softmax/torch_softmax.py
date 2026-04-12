import torch
import time
import matplotlib.pyplot as plt

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda")

# Sizes to test
sizes = [512, 1024, 2048, 4096]
times = []

# Number of repetitions for stable timing
repeats = 100

for size in sizes:
    # Create random tensor on GPU
    x = torch.randn(size, size, device=device)

    # Warm-up (important for GPU timing)
    for _ in range(10):
        _ = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(repeats):
        y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / repeats * 1000
    times.append(avg_time_ms)

    print(f"Size {size}x{size}: {avg_time_ms:.4f} ms")

# Plot
plt.figure()
plt.plot(sizes, times, marker='o')
plt.xlabel("Tensor Size (N x N)")
plt.ylabel("Average Softmax Time (ms)")
plt.title("GPU Softmax Performance (PyTorch)")
plt.grid(True)
plt.show()
