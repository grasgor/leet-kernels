#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; //columns
    int y = blockDim.y * blockIdx.y + threadIdx.y; //rows
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    int temp = 0;
    for(int i = x; i < M; i += stride_x){
        for(int j = y; j < N; j += stride_y){
            temp += (input[j * M + i] == K) ? 1 : 0;
        }
    }
    atomicAdd(output, temp);

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}