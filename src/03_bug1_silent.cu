// 03_bug1_silent.cu -- BUG HUNT 1: "It runs, there is no error, but the answer is all zeros."
//
// Goal of the program: out[i] = i * i
//
// Build:  make -C ..        (or: nvcc -g -o 03_bug1_silent 03_bug1_silent.cu)
// Run:    ./03_bug1_silent
//
// STEP 1. Run it. Expected "out[3] = 9". What do you get? Any error message?
// STEP 2. CUDA calls do not stop your program when they fail. They RETURN an error code and
//         it is your job to look at it, like this:   CHECK(cudaMalloc(&d_out, bytes));   A kernel
//         launch returns nothing, so ask right after it: remove the "//" in front of the CHECK line.
// STEP 3. Rebuild, run, read the message. Hint: a block may hold at most 1024 threads.
#include <cstdio>
#include <cstdlib>

// Print a message and stop if a CUDA call failed. ("do { } while (0)" and the "\" are how a multi-line macro is written.)
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
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_out, 0, bytes);

    int threads = 2048;                         // "more threads per block must be faster, right?"
    int blocks  = (n + threads - 1) / threads;
    square<<<blocks, threads>>>(d_out, n);
    // CHECK(cudaGetLastError());               // <-- STEP 2: did the launch itself fail?
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("out[3]    = %g   (expected 9)\n", out[3]);
    printf("out[1000] = %g   (expected 1e+06)\n", out[1000]);

    cudaFree(d_out);
    free(out);
    return 0;
}
