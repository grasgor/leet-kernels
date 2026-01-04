## Establishing baseline

In a naive softmax kernel of size N x N
We have 5N mathematical operations and 8N memory ops so the arithmetic intensity is 5/8 = 0.625

Now, here are our hardware specs
```bash
Device 0: NVIDIA GeForce RTX 5060 Laptop GPU
  Compute capability: 13.0
  Total global memory: 8.08 GB
  Shared memory per SM: 100.00 KB
  L2 cache size: 32.00 MB
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 1536
  Number of multiprocessors (SMs): 26
  Core clock: 1.46 GHz
  Memory clock: 12001.00 MHz
  Memory bus width: 128 bits
  Estimated peak memory bandwidth: 384.03 GB/s
  CUDA cores per SM: 128
  Total CUDA cores: 3328
  Estimated FP32 peak: 9.68 TFLOP/s
-------------------------------------------------
```

The peak memory bandwith is 384.03 GB/s and peak FP32 FLOP/s are 9.68 TFLOP/s

threshold ​= 9.68 × 10^12 / 384.03×10^9 ​ ≈ 25.2 FLOPs/byte

0.625 ≪ 25.2 so the kernel is heavily memory-bound.

Our roofline would be 0.625 * 384.03 = 240.018 GFLOP/s

## Kernels so far
All the kernels make use of online softmax
  - first kernel: online + accessed global memory twice
  - second kernel: online + accessed global memory once but introduces shared memory
  - third kernel: online + accessed global memory once + intra warp reduction helps in reducing size of shared mem + speed ups using only warp primitives





## Design choices learnt from the block reduction kernel

```bash
# use early exit rather than if condition with a __syncthread() to avoid deadlocks
if (row >= M) return;

# instead of 
if (row < M){
  ... core logic
  __syncthreads(); # --> can cause deadlock
}


# prefer using pointer arithmetic instead of explicit indexing
const unsigned int row = blockIdx.x;
float* dinput = input_tensor + row * N; # use this 
float element = dinput[i];

# rather than using
float element = input_tensor[row * N + i];

# a. you avoid recalculating index over and over again
# b. pointer arithmetic was seemingly 0.2ms faster on a 4098 x 4098 input than explicit indexing (idk if that is how it should be)
```