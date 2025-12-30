#include <cuda_runtime.h>

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


__global__ void softmax_block_reduction(float* input_tensor, const int M, const int N){
    // iterative improvement over online softmax 
    // each block solves one row ie if blockDim.x = 32, then 32 threads solve the entire row instead of single thread

    const unsigned int row = blockIdx.x;
    const unsigned int tid_x = threadIdx.x;
    __shared__ float smem[1024];
    if(row >= M) return;

    float local_max = -INFINITY;
    float local_norm = 0.0f;
    for(int i = tid_x; i<N; i += blockDim.x){
        float x = input_tensor[row * N + i];
        if(x > local_max){
            local_norm *= __expf(local_max - x);
            local_max = x;
        }
        local_norm += __expf(x - local_max);
    }
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
    float row_max = smem[0];

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

    __syncthreads();

    // finally, compute softmax
    for (int i = tid_x; i < N; i += blockDim.x) {
        int element_idx = row * N + i; 
        input_tensor[element_idx] = expf(input_tensor[element_idx] - row_max) / row_norm;
    }
}