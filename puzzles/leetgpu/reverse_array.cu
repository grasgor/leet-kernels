#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
int threadId = blockDim.x * blockIdx.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for(int i = threadId; i < N/2; i+=stride){
    float temp = input[i];
    input[i] = input[N-i-1];
    input[N-i-1] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}