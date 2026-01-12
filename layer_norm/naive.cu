#include <cuda_runtime.h>

__global__ void layer_norm(float* input_tensor, float* output, const int M, const int N, const float eps){
    //assumption: tensor is stored in row major layout

    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_id >= M) return;

    float* row = input_tensor + row_id * N;
    float* output_row = output + row_id * N;

    //first pass to calculate row mean
    float row_mean = 0.0f;
    for(int idx = 0; idx < N; idx++){
        row_mean += row[idx];
    }
    row_mean /= N;

    //second pass to calculate variance (sigma square)
    float row_variance = 0.0f;
    for(int idx = 0; idx < N; idx++){
        float diff = (row_mean - row[idx]);
        row_variance += diff * diff; 
    }
    row_variance /= N;

    float inv_std = 1/sqrtf(row_variance + eps);
    //third pass to actually modify each element
    for(int idx = 0; idx < N; idx++){
        output_row[idx] = (row[idx] - row_mean) * inv_std;
    }

}