#include <cuda_runtime.h>
#include "include/softmax.cuh"


__global__ void online_softmax(float* input_tensor, const int M, const int N){
    //numerically stable online_softmax but non coalesced
    // M -> num rows
    // N -> num columns
    // assumption that the tensor is stored in a row major layout
    // one thread takes care of one row

    //[3, 5, 4]

    const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M){
        float row_max = -INFINITY;
        float norm = 0.0f;
        for(int i = 0; i<N; i++){
            int row_idx = row * N + i;

            float x = input_tensor[row_idx];
            if (x > row_max){
                norm *= __expf(row_max - x);
                row_max = x;  
            }
            norm += __expf(x - row_max);
        }

        for(int i = 0; i<N; i++){
            int row_idx = row * N + i;
            float x = input_tensor[row_idx];
            input_tensor[row_idx] = __expf(x - row_max) / norm;
        }
    }
}
