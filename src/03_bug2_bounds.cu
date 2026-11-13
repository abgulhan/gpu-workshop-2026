// 03_bug2_bounds.cu -- BUG HUNT 2: "The answer looks fine. Is the code correct?"
//
// Goal of the program: smooth a noisy signal. Each output value is the average of
// an input value and its two neighbours:  out[i] = (in[i-1] + in[i] + in[i+1]) / 3
//
// Build:  nvcc -g -lineinfo -o 03_bug2_bounds 03_bug2_bounds.cu
// Run:    ./03_bug2_bounds
//
// STEP 1. Run it. Every CUDA call is checked and no error is reported.
//         But look at the first and last output values. Every input is between
//         10 and 11, so every average should be too. Are they?
// STEP 2. Run it under the memory checker, a tool that watches every read and write:
//             compute-sanitizer ./03_bug2_bounds
//         Read the report: which LINE, which THREAD, which BLOCK, read or write?
// STEP 3. Work out which element i that thread was handling, and fix the kernel.
#include <cstdio>
#include <cstdlib>

#define CHECK(call)  /* print a message and stop if a CUDA call failed */    \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void smooth(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3.0f;
}

int main()
{
    int n = 1000;
    size_t bytes = (size_t)n * sizeof(float);
    float *in  = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) in[i] = 10.0f + (float)(rand() % 100) / 100.0f;   // 10 + noise

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    smooth<<<blocks, threads>>>(d_in, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("in : %.3f %.3f %.3f ... %.3f %.3f\n", in[0], in[1], in[2], in[n - 2], in[n - 1]);
    printf("out: %.3f %.3f %.3f ... %.3f %.3f\n", out[0], out[1], out[2], out[n - 2], out[n - 1]);
    printf("No CUDA errors reported.\n");

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(in); free(out);
    return 0;
}
