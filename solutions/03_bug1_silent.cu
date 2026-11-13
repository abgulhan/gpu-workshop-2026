// 03_bug1_silent.cu -- SOLUTION.
//
// With the checks in place the program stops with
//     CUDA error at src/03_bug1_silent.cu:47: invalid configuration argument
// (CUDA 13 words it "invalid argument").
// A block can hold at most 1024 threads. 2048 is not allowed, so the kernel never ran
// and the output stayed at zero. Fix: threads = 256.
//
// Lesson: check the return value of EVERY CUDA call, and call cudaGetLastError()
// after every kernel launch.
#include <cstdio>
#include <cstdlib>

// Print a readable message and stop if a CUDA call failed.
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void square(float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (float)i * (float)i;
}

int main()
{
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);
    float *out = (float *)calloc(n, sizeof(float));

    float *d_out;
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemset(d_out, 0, bytes));

    int threads = 256;                          // FIX: at most 1024 threads per block
    int blocks  = (n + threads - 1) / threads;
    square<<<blocks, threads>>>(d_out, n);
    CHECK(cudaGetLastError());                  // did the launch fail?
    CHECK(cudaDeviceSynchronize());             // did the kernel fail while running?

    CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("out[3]    = %g   (expected 9)\n", out[3]);
    printf("out[1000] = %g   (expected 1e+06)\n", out[1000]);

    CHECK(cudaFree(d_out));
    free(out);
    return 0;
}
