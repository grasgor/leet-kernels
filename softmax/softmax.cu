#include <cuda_runtime.h>

__global__ void naive_softmax(float* input_tensor, float* output_tensor, const int M, const int N){
    //numerically stable naive_softmax but non coalesced
    // M -> num rows
    // N -> num columns
    // assumption that the tensor is stored in a row major layout


}