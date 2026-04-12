#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void SumReduction(const float* input, float* output, int N){
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int stride = gridDim.x * blockDim.x;

    //grid-stride accumulation
    float sum = 0.0f;
    for (int i = gid; i < N; i += stride){
        sum += input[i];
    }

    //warp-level reduction
    sum = warpReduceSum(sum);

    //one atomic per warp
    if ((tid & 31) == 0){
        atomicAdd(output, sum);
    }
}

extern "C" void solve(const float* input, float* output, int N){
    const int BLOCK = 256;
    const int GRID  = 256;

    cudaMemset(output, 0, sizeof(float));
    SumReduction<<<GRID, BLOCK>>>(input, output, N);
}
