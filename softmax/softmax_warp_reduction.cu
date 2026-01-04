#include <cuda_runtime.h>
#include "include/softmax.cuh"


__device__ __forceinline__ float warpReduceMax(const int warp_size, float val){
    /** 
        Consider a warp with 32 threads. Each thread in the warp has its local max
        warpReduceMax takes a thread val and compares with thread val at offset ie fmaxf(warp[0].val, warp[0 + offset].val)
        same logn reduction strategy but now at warp level
    **/
    for(int offset = warp_size / 2; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

//modeled after warpReduceMax
__device__ __forceinline__ float warpReduceSum(const int warp_size, float val){
    for(int offset = warp_size / 2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_warp_reduction(float* input_tensor, float* output_tensor, const int M, const int N){
    //iterative improvement over block reduction

    const unsigned int row = blockIdx.x;
    const unsigned int tid_x = threadIdx.x;
    const unsigned int warp_size = 32;
    
    //pointer arithemtic to move input row forward one row at a time
    float* dinput = input_tensor + row * N;
    float *doutput = output_tensor + row * N;

    extern __shared__ float smem[];

    if (row >= M) return;

    //thread register variables for local_max and local_norm
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    //core online softmax logic to calculate max and norm in single pass
    for(int i = tid_x; i < N; i += blockDim.x){
        float curr_x = dinput[i];
        if(curr_x > local_max){
            local_norm *= __expf(local_max - curr_x);
            local_max = curr_x;
        }
        local_norm += __expf(curr_x - local_max);
    }

    //warp reduced local_max
    float per_warp_max = warpReduceMax(warp_size, local_max);

    int lane_id = tid_x % warp_size;
    int warp_id = tid_x / warp_size;

    //block reduction will take place if blockDim.x > warp_size
    
    if(blockDim.x > warp_size){
        //warp leaders write to smem
        if (tid_x % warp_size == 0) {
            // tid_x / 32 = warp_id
            // tid_x % 32 = lane_id
            smem[tid_x / warp_size] = per_warp_max;
        }
        //sync threads after writing to smem
        __syncthreads();
    
        //now perform block reduction but only using warp primitives
        //we only us the threads of the first warp
        if(tid_x < warp_size){
            //since we have to perform block reduction using warp primitives, load smem values to first warp thread registers
            //local_max was the local variable for storing in thread registers
            per_warp_max = (tid_x < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid_x] : -INFINITY;
            
            //perform warp reduction
            float this_warp_max = warpReduceMax(warp_size, per_warp_max);

            if (tid_x == 0){
                smem[0] = this_warp_max;
            }
        }
    } 

    else{
        //else if blockDim.x < warp_size then we have already reduced since there is only one warp and warp leader thread already has the max val
        if (tid_x == 0){
                smem[0] = per_warp_max;
        }
    }
    __syncthreads();

    float row_max = smem[0];


    //perform local norm correction with row_max before aggregating 
    local_norm *= __expf(local_max - row_max);
    float per_warp_norm = warpReduceSum(warp_size, local_norm);

    //same logic as reduction for row max
    if(blockDim.x > warp_size){
        //warp leaders first write to smem
        if(tid_x % warp_size == 0){
            //tid_x % warp_size == lane_id
            //tid_x / warp_size == warp_id
            smem[tid_x / warp_size] = per_warp_norm;
        }
        __syncthreads();

        //use first warp to perform block reduction
        if(tid_x < warp_size){
            //load data back from smem into first warp threads' registers
            per_warp_norm = (tid_x < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid_x] : 0.0f;
            float this_warp_norm = warpReduceSum(warp_size, per_warp_norm);

            if(tid_x == 0){
                smem[0] = this_warp_norm;
            }
        }
    }
    else{
        if(tid_x == 0){
            smem[0] = per_warp_norm;
        }
    }
    __syncthreads();

    float row_norm = smem[0];

    //final pass to then actually compute softmax row-wise
    for(int i = tid_x; i < N; i += blockDim.x){
        doutput[i] = __expf(dinput[i] - row_max) / row_norm;
    }
}