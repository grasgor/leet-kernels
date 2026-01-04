#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void online_softmax(float* input_tensor, const int M, const int N);

__global__ void softmax_block_reduction(float* input_tensor, const int M, const int N);

__global__ void softmax_warp_reduction(float* input_tensor, float* output_tensor, const int M, const int N);