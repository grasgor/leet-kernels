#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "include/softmax.cuh"

// Helper to check tensor constraints
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor dispatch_naive_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    torch::Tensor output = input.clone();
    
    int M = input.size(0);
    int N = input.size(1);
    
    dim3 block(256);
    dim3 grid(CEIL_DIV(M, 256));
    
    naive_softmax_3pass<<<grid, block>>>(output.data_ptr<float>(), M, N);
    
    return output;
}

torch::Tensor dispatch_online_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    // Clone input because online_softmax is in-place
    torch::Tensor output = input.clone();
    
    int M = input.size(0);
    int N = input.size(1);
    
    // Launch configuration
    // Original kernel online_softmax uses:
    // row = blockDim.x * blockIdx.x + threadIdx.x
    // It seems one thread takes care of one row?
    // Let's re-read the kernel source briefly from memory.
    // "const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;"
    // "if(row < M) ..."
    // So yes, one thread per row. 
    
    dim3 block(256);
    dim3 grid(CEIL_DIV(M, 256));
    
    online_softmax<<<grid, block>>>(output.data_ptr<float>(), M, N);
    
    return output;
}

torch::Tensor dispatch_block_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    torch::Tensor output = input.clone();

    int M = input.size(0);
    int N = input.size(1);

    // blocks correspond to rows
    // threads correspond to elements in row (up to blockDim.x)
    // "softmax_block_reduction(float* input_tensor, const int M, const int N)"
    // "const unsigned int row = blockIdx.x;"
    // "for(int i = tid_x; i<N; i += blockDim.x)"
    // So 1 block per row.
    
    // Max threads per block is 1024
    int threads = 1024; // Use max threads for block reduction 
    if (N < 1024) threads = N; // Should be power of 2 for reduction? 
    // The kernel implementation assumes power of 2 reduction:
    // "for(int stride = blockDim.x / 2; stride > 0; stride /= 2)"
    // So we should pick next power of 2 or just stick to 1024/512/256 etc?
    // It's safer to use a power of 2 for blockDim.x. 
    // Let's stick to 1024 as generic "large enough" or min(NextPowerOf2(N), 1024).
    // For simplicity let's use 1024. If N < 1024, it loops correctly?
    // "for(int i = tid_x; i<N; i += blockDim.x)" -> handles N < blockDim.x
    // Reduction loop: "stride = blockDim.x / 2".
    // If blockDim.x is not power of 2, stride /= 2 might behave oddly if odd.
    // So we MUST ensure blockDim.x is power of 2.
    
    threads = 1024;
    while(threads > N && threads > 32) threads /= 2;

    dim3 block(threads);
    dim3 grid(M);

    softmax_block_reduction<<<grid, block>>>(output.data_ptr<float>(), M, N);

    return output;
}

torch::Tensor dispatch_warp_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    // softmax_warp_reduction is NOT in-place based on signature:
    // void softmax_warp_reduction(float* input_tensor, float* output_tensor, const int M, const int N)
    
    int M = input.size(0);
    int N = input.size(1);
    
    torch::Tensor output = torch::empty_like(input);

    // "const unsigned int row = blockIdx.x;"
    // So 1 block per row again.
    
    int threads = 1024;
    while(threads > N && threads > 32) threads /= 2;
    // Kernel uses warp shuffling, so threads must be multiple of 32 for safety?
    // Code says: "const unsigned int warp_size = 32;"
    // "int warp_id = tid_x / warp_size;"
    // So threads should be at least 32.

    dim3 block(threads);
    dim3 grid(M);
    
    // Dynamic shared memory request
    // "extern __shared__ float smem[];"
    // Used for:
    // "smem[tid_x / warp_size] = per_warp_max;" => size needed: threads/32
    // then "smem[0] = this_warp_max;"
    
    // Actually the kernel does:
    // if(blockDim.x > warp_size) ...
    //   smem[tid_x / warp_size] = ...
    // max usage is related to number of warps.
    // Threads=1024 -> 32 warps -> 32 floats.
    // Size is small. verify bytes.
    size_t smem_size = (threads / 32) * sizeof(float);
    
    softmax_warp_reduction<<<grid, block, smem_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_naive_softmax", &dispatch_naive_softmax, "Naive 3-Pass Softmax");
    m.def("dispatch_online_softmax", &dispatch_online_softmax, "Online Softmax");
    m.def("dispatch_block_softmax", &dispatch_block_softmax, "Block Reduction Softmax");
    m.def("dispatch_warp_softmax", &dispatch_warp_softmax, "Warp Reduction Softmax");
}
