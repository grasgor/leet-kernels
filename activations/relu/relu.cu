#include <cstdio>
#include <cuda_runtime.h>


#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (error code: %d), at %s:%d\n", \
                    cudaGetErrorString(err_), err_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void relu_float4(float* input_tensor, float* output_tensor, const int M, const int N){
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    //since matrix is stored in row major format, we can ensure coalesced
    //global memory access if threads with consecutive x iterate along the columns
    const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // we will use float4 vectorization
    float4* input_float4 = reinterpret_cast<float4*>(input_tensor);
    float4* output_float4 = reinterpret_cast<float4*>(output_tensor);
    
    //float4 loads 4 elements at once so the iteration across columns N
    //is reduced by 4 times
    int N4 = N / 4;
    if(row < M && col < N4){
        int idx = row * N4 + col;
        float4 u = input_float4[idx];

        // apply relu
        //for leaky_relu = fmaxf(alpha * u.x, u.x);
        u.x = fmaxf(0.0f, u.x);
        u.y = fmaxf(0.0f, u.y);
        u.z = fmaxf(0.0f, u.z);
        u.w = fmaxf(0.0f, u.w);

        output_float4[idx] = u;        
    }

    //in each row the tail starts at N4
    int tail_start = N4 * 4;
    int rem = N % 4;
    //if num columns are odd, there will be tail elements in each row
    if(row < M && col == 0 && rem != 0){
        //we make a single thread ie the one at index (row, 0)
        for(int j = tail_start; j < N; j++){
            int rem_idx = row * N + j;
            float u = input_tensor[rem_idx];
            u = fmaxf(0.0f, u);
            output_tensor[rem_idx] = u; 
        }
    }
}

int main(){

    int N = 1024;
    size_t fp32_size = N * N * sizeof(float);
    
    //assign float pointers
    //on host
    float *input_tensor, *output_tensor;
    
    //on device
    float *dinput, *doutput;

    //allocate memory on host
    input_tensor = new float[N * N];
    output_tensor = new float[N * N];

    //allocate data to memory on host (cpu)
    for(int i = 0; i<N; i++){
        for(int j = 0; j < N; j++){
            if(i == j){
                input_tensor[i * N + j] = 4.5f;
            }
            else{ 
                input_tensor[i * N + j] = -1.0f;
            }
        }        
    }

    
    //allocate memory on device (gpu)
    CUDA_CHECK(cudaMalloc((void**)&dinput, fp32_size));
    CUDA_CHECK(cudaMalloc((void**)&doutput, fp32_size));


    //move data from host to allocated device memory
    CUDA_CHECK(cudaMemcpy(dinput, input_tensor, fp32_size, cudaMemcpyHostToDevice));


    int N4 = N/4;
    //define grid, block, thread specs
    dim3 block(32, 32);
    dim3 grid((block.x + N4 - 1)/ block.x, (block.y + N - 1)/ block.y);


    // Timing using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    //launch kernel
    relu_float4<<<grid, block>>>(dinput, doutput, N, N);

    // check kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    //move data from device to host
    CUDA_CHECK(cudaMemcpy(output_tensor, doutput, fp32_size, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (i == j) ? 4.5f : 0.0f;
            if (output_tensor[i * N + j] != expected) {
                correct = false;
                printf("Mismatch at (%d,%d): got %f, expected %f\n", i, j, output_tensor[i * N + j], expected);
                // optionally break here to avoid flooding
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        printf("Output is correct: diagonals = 4.5, others = 0\n");
    } else {
        printf("Output is incorrect!\n");
    }

    //clear memory
    delete[] input_tensor;
    delete[] output_tensor;
    cudaFree(dinput);
    cudaFree(doutput);
}