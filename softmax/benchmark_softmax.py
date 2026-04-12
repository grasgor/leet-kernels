import os
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

# --- Compile and Load Extension ---
print("Compiling and loading CUDA kernels...")

# Explicitly list all source files
sources = [
    'softmax_wrapper.cu',
    'naive_3_pass.cu',
    'softmax_no_reduction.cu',
    'softmax_block_reduction.cu',
    'softmax_warp_reduction.cu'
]

# Using absolute paths to be safe
base_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(base_dir, s) for s in sources]

softmax_cuda = load(
    name='softmax_cuda',
    sources=sources,
    # include_dirs=[base_dir],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

print("Kernels loaded successfully.")

# --- Benchmarking Setup ---

def benchmark(func, input_tensor, repeats=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        _ = func(input_tensor)
    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    for _ in range(repeats):
        _ = func(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / repeats * 1000  # ms

def check_correctness(inputs, ref_func, cust_func, kernel_name):
    ref = ref_func(inputs)
    cust = cust_func(inputs)
    
    # Tolerances might need adjustment for fast math
    if torch.allclose(ref, cust, atol=1e-3, rtol=1e-3):
        print(f"[{kernel_name}] Correctness check passed.")
    else:
        diff = (ref - cust).abs().max().item()
        print(f"[{kernel_name}] Correctness check FAILED. Max diff: {diff:.6f}")

# Sizes to test (NxN)
sizes = [512, 1024, 2048, 4096, 8192]
device = torch.device('cuda')

results = {
    'PyTorch': [],
    'Naive (3 Pass)': [],
    'Online (No Reduction)': [],
    'Block Reduction': [],
    'Warp Reduction': []
}

# --- Main Loop ---

for N in sizes:
    print(f"\nBenchmarking size {N}x{N}...")
    x = torch.randn(N, N, device=device, dtype=torch.float32)
    
    # PyTorch Native
    # Use dim=-1 to match kernels which operate row-wise
    torch_func = lambda t: torch.softmax(t, dim=-1)
    
    
    # Kernels
    naive_func = softmax_cuda.dispatch_naive_softmax
    online_func = softmax_cuda.dispatch_online_softmax
    block_func = softmax_cuda.dispatch_block_softmax
    warp_func = softmax_cuda.dispatch_warp_softmax
    
    # Check correctness on first size only to save time/spam, 
    # OR check on all if critical. Let's check on all for robust verification.
    if N <= 1024:
        check_correctness(x, torch_func, naive_func, "Naive")
        check_correctness(x, torch_func, online_func, "Online")
        check_correctness(x, torch_func, block_func, "Block")
        check_correctness(x, torch_func, warp_func, "Warp")

    # Benchmark
    results['PyTorch'].append(benchmark(torch_func, x))
    results['Naive (3 Pass)'].append(benchmark(naive_func, x))
    results['Online (No Reduction)'].append(benchmark(online_func, x))
    results['Block Reduction'].append(benchmark(block_func, x))
    results['Warp Reduction'].append(benchmark(warp_func, x))
    
    print(f"PyTorch: {results['PyTorch'][-1]:.4f} ms")
    print(f"Naive:   {results['Naive (3 Pass)'][-1]:.4f} ms")
    print(f"Online:  {results['Online (No Reduction)'][-1]:.4f} ms")
    print(f"Block:   {results['Block Reduction'][-1]:.4f} ms")
    print(f"Warp:    {results['Warp Reduction'][-1]:.4f} ms")

# --- Plotting ---

plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(sizes, times, marker='o', label=name)

plt.xlabel('Tensor Size (N x N)')
plt.ylabel('Latency (ms)')
plt.title('Softmax Kernel Performance Benchmark')
plt.legend()
plt.grid(True)
plt.yscale('log') # Log scale might be useful if differences are huge
plt.savefig('softmax_benchmark.png')
print("\nBenchmark complete. Plot saved to softmax_benchmark.png")
