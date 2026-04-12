#include <cuda_runtime.h>
#include "include/softmax.cuh"

__global__ void naive_softmax_3pass(float* input_tensor, const int M, const int N){
    // naive numerically stable softmax (3 pass, non coalesced)
    // M -> num rows
    // N -> num columns
    // assumption that the tensor is stored in a row major layout
    // one thread takes care of one row

    const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M){

        float row_max = -INFINITY;

        // pass 1: find max
        for(int i = 0; i < N; i++){
            int row_idx = row * N + i;
            float x = input_tensor[row_idx];
            if(x > row_max){
                row_max = x;
            }
        }

        float norm = 0.0f;

        // pass 2: compute normalization term
        for(int i = 0; i < N; i++){
            int row_idx = row * N + i;
            float x = input_tensor[row_idx];
            norm += __expf(x - row_max);
        }

        // pass 3: write output
        for(int i = 0; i < N; i++){
            int row_idx = row * N + i;
            float x = input_tensor[row_idx];
            input_tensor[row_idx] = __expf(x - row_max) / norm;
        }
    }
}
