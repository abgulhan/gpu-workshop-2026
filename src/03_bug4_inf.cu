// 03_bug4_inf.cu -- BUG HUNT 4: "My result is inf. Which of my 4096 threads did it?"
//
// Goal of the program: for two sets of measurements, compute the ratio
// after / before for every sample, then report the mean ratio.
//
// Build (debug build: -g for the CPU side, -G for the GPU side):
//         nvcc -g -G -o 03_bug4_inf 03_bug4_inf.cu
// Run:    ./03_bug4_inf
//
// STEP 1. Run it. The mean ratio should be about 2. What do you get?
// STEP 2. Use the GPU debugger to stop INSIDE the kernel and look at the data:
//             cuda-gdb ./03_bug4_inf
//             (cuda-gdb) break 03_bug4_inf.cu:51     stop on the line that divides
//             (cuda-gdb) run
//             (cuda-gdb) info cuda kernels           what is running on the GPU?
//             (cuda-gdb) info cuda threads           where is every GPU thread?
//             (cuda-gdb) print i                     which element does this thread own?
//             (cuda-gdb) cuda block 2 thread 5       jump to another thread
//             (cuda-gdb) print i
//             (cuda-gdb) print before[i]
//         Now let the debugger find the culprit for you: stop on the NEXT line,
//         but only in a thread whose result r is huge:
//             (cuda-gdb) break 03_bug4_inf.cu:52 if r > 1000000
//             (cuda-gdb) continue
//             (cuda-gdb) cuda kernel block thread    which thread stopped?
//             (cuda-gdb) print i
//             (cuda-gdb) print r
//             (cuda-gdb) print before[i]
//             (cuda-gdb) print after[i]
//             (cuda-gdb) quit                        answer y when it asks
//         No time to type? Run almost the same session from a script:
//             cuda-gdb -batch -x 03_bug4_inf.gdb ./03_bug4_inf
// STEP 3. Decide what the program should do with that sample, and fix it.
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

__global__ void ratio_kernel(const float *before, const float *after, float *ratio, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = after[i] / before[i];
        ratio[i] = r;
    }
}

int main()
{
    int n = 4096;
    size_t bytes = (size_t)n * sizeof(float);
    float *before = (float *)malloc(bytes);
    float *after  = (float *)malloc(bytes);
    float *ratio  = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        before[i] = 1.0f + (float)(i % 10);
        after[i]  = 2.0f * before[i];
    }
    before[1234] = 0.0f;            // one bad measurement hiding in the data

    float *d_before, *d_after, *d_ratio;
    CHECK(cudaMalloc(&d_before, bytes));
    CHECK(cudaMalloc(&d_after, bytes));
    CHECK(cudaMalloc(&d_ratio, bytes));
    CHECK(cudaMemcpy(d_before, before, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_after, after, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    ratio_kernel<<<blocks, threads>>>(d_before, d_after, d_ratio, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(ratio, d_ratio, bytes, cudaMemcpyDeviceToHost));

    double sum = 0;
    for (int i = 0; i < n; i++) sum += ratio[i];
    printf("mean ratio = %g   (expected about 2)\n", sum / n);

    CHECK(cudaFree(d_before));
    CHECK(cudaFree(d_after));
    CHECK(cudaFree(d_ratio));
    free(before); free(after); free(ratio);
    return 0;
}
