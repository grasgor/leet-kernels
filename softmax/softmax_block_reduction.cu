#include <cuda_runtime.h>
#include "include/softmax.cuh"

__global__ void softmax_block_reduction(float* input_tensor, const int M, const int N){
    // iterative improvement over online softmax //online softmax that uses block level reductions
    // each block solves one row ie if blockDim.x = 32, then 32 threads solve the entire row instead of single thread

    const unsigned int row = blockIdx.x;
    const unsigned int tid_x = threadIdx.x;

    //static allocation of shared memory
    __shared__ float smem[1024];

    if(row >= M) return; //allows early exit

    //local_max and local_norm are thread specific register variables
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    //online softmax with local_norm error correction
    for(int i = tid_x; i<N; i += blockDim.x){
        float x = input_tensor[row * N + i];
        if(x > local_max){
            local_norm *= __expf(local_max - x);
            local_max = x;
        }
        local_norm += __expf(x - local_max);
    }
    //save thread register local_max value in smem
    smem[tid_x] = local_max;
    __syncthreads();
    
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid_x < stride){
            //block reduction in the order of log(n)
            //instead of doing a bubble sort like comparision for finding the max which would take O(n)
            //we compare each element with its corresponding in the other half of the array and then iteratively half the search space
            smem[tid_x] = fmaxf(smem[tid_x], smem[tid_x + stride]);
        }
        __syncthreads();
    }
    //first element is row_max after reduction
    float row_max = smem[0];

    //load thread register into same smem (shared memory reused, avoids creating two instances for local_max as well as local_norm)
    smem[tid_x] = local_norm * __expf(local_max - row_max);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid_x < stride){
            //block reduction in the order of log(n)
            //instead of doing a bubble sort like comparision for finding the max which would take O(n)
            //we compare each element with its corresponding in the other half of the array and then iteratively half the search space
            smem[tid_x] += smem[tid_x + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];

    // final and second pass to actually compute softmax
    for (int i = tid_x; i < N; i += blockDim.x) {
        int element_idx = row * N + i; 
        input_tensor[element_idx] = __expf(input_tensor[element_idx] - row_max) / row_norm;
    }
}
