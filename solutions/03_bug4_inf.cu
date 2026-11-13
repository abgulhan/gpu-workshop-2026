// 03_bug4_inf.cu -- SOLUTION.
//
// cuda-gdb stops in block 4, thread 210: i = 4*256 + 210 = 1234, before[i] = 0.
// after[i] / 0 = inf, and one inf makes the whole mean inf.
// Fix used here: mark a sample with no valid "before" value as missing (ratio = -1)
// and leave it out of the mean. What the right fix is depends on your science;
// the point is that the debugger told you WHICH sample and WHY.
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

__global__ void ratio_kernel(const float *before, const float *after, float *ratio, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (before[i] == 0.0f)
        ratio[i] = -1.0f;                       // FIX: flag the sample as missing
    else
        ratio[i] = after[i] / before[i];
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
    int used = 0;
    for (int i = 0; i < n; i++)
        if (ratio[i] >= 0) { sum += ratio[i]; used++; }
    printf("mean ratio = %g   (expected about 2; %d of %d samples used)\n", sum / used, used, n);

    CHECK(cudaFree(d_before));
    CHECK(cudaFree(d_after));
    CHECK(cudaFree(d_ratio));
    free(before); free(after); free(ratio);
    return 0;
}
