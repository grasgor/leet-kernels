#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; //column
    int j = blockDim.y * blockIdx.y + threadIdx.y; //row
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    for(int idx = j; idx < rows; idx += stride_y){
        for(int k = i; k < cols; k += stride_x){
            output[k * rows + idx] = input[idx * cols + k]; // row_num * width + col_num = i, j
        }
    }
}
//     while (x < cols && y < rows){
//         output[y * cols + x] = input
//     }
// }

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}