#include <cuda_runtime.h>

//swish and silu are the same
__global__ void swish(float* input_tensor, float* output_tensor, const int M, const int N){
    // Swish(x)= x*sigmoid(βx)
    // SiLU is the standardized name wherein β = 1
    const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    float4* input4 = reinterpret_cast<float4*>(input_tensor);
    float4* output4 = reinterpret_cast<float4*>(output_tensor);

    int N4 = N / 4;

    if(row < M && col < N4){
        int idx = row * N4 + col;
        float4 val = input4[idx];
        val.x = val.x / (1.0f + __expf(-val.x));
        val.y = val.y / (1.0f + __expf(-val.y));
        val.z = val.z / (1.0f + __expf(-val.z));
        val.w = val.w / (1.0f + __expf(-val.w));
        output4[idx] = val;
    }

    int tail_idx = N4 * 4;
    int rem = N % 4;
    if(row < M && col == 0 && rem != 0){
        for(int j = tail_idx; j<N; j++){
            int rem_idx = row * N + j;
            float element = input_tensor[rem_idx];
            output_tensor[rem_idx] =  element / (1.0f + __expf(-element));
        }
    }
}   
