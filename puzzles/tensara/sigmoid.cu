#include <cuda_runtime.h>

__global__ void sigmoid_2d(const float* input, float* output, size_t n, size_t m){
    //row-major 
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    int n4 = n/4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    if(row < m && col < n4){
        int idx = row*n4 + col;
        float4 v = input4[idx];
        float4 out = {
            1.f / (1.f + __expf(-v.x)),
            1.f / (1.f + __expf(-v.y)),
            1.f / (1.f + __expf(-v.z)),
            1.f / (1.f + __expf(-v.w)),
        };
        output4[idx] = out;
    }

    //tail handling per row
    int tailStart = n4 * 4;

    if(row < m && col < (n - tailStart)){
        int idx = row*n + tailStart + col;

        output[idx] = 1.f / (1.f + __expf(-input[idx]));
    }
}

__global__ void sigmoid_1d(const float* input, float* output, size_t n, size_t m){
    size_t N = m * n;
    size_t N4 = N / 4;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = tid; i < N4; i += stride){
        float4 v = input4[i];

        float4 out = {
            1.f / (1.f + __expf(-v.x)),
            1.f / (1.f + __expf(-v.y)),
            1.f / (1.f + __expf(-v.z)),
            1.f / (1.f + __expf(-v.w))
        };

        output4[i] = out;
    }

    int tailStart = N4 * 4;
    if(tid == 0){
        for(int i = tailStart; i<N; i++){
            output[i] = 1.f / (1.f + __expf(-input[i]));
        }
    }
}

// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    size_t total = m * n;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    // prevent excessive grid launch
    gridSize = min(gridSize, 65535);

    // sigmoid_1d<<<gridSize, blockSize>>>(input, output, n, m);

    int n4 = n / 4;
    dim3 block(32, 8);
    dim3 grid(
        (n4 + block.x - 1) / block.x,
        (m  + block.y - 1) / block.y
    );
    sigmoid_2d<<<grid, block>>>(input, output, n, m);
}