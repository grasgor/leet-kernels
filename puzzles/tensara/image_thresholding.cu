#include <cuda_runtime.h>

__global__ void threshold(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    const float4* input4 = reinterpret_cast<const float4*>(input_image);
    float4* output4 = reinterpret_cast<float4*>(output_image);

    size_t num_elements = height * width;
    size_t head = num_elements / 4;
    size_t tail = num_elements % 4;

    if (x < head) {
        float4 val = input4[x];
        val.x = (val.x > threshold_value) ? 255.0f : 0.0f;
        val.y = (val.y > threshold_value) ? 255.0f : 0.0f;
        val.z = (val.z > threshold_value) ? 255.0f : 0.0f;
        val.w = (val.w > threshold_value) ? 255.0f : 0.0f;
        output4[x] = val;
    }

    if (tail > 0 && x == 0) {
        const size_t start = head * 4;
        for (size_t i = start; i < num_elements; ++i) {
            output_image[i] = (input_image[i] > threshold_value) ? 255.0f : 0.0f;
        }
    }
}

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    size_t num_pixels = height * width;
    size_t num_vec4 = num_pixels / 4;

    int threadsPerBlock = 1024;
    int numBlocks = (num_vec4 + threadsPerBlock - 1) / threadsPerBlock;

    threshold<<<numBlocks, threadsPerBlock>>>(input_image, threshold_value, output_image, height, width);
}
