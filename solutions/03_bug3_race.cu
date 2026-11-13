// 03_bug3_race.cu -- SOLUTION.
//
// "*count = *count + 1" is three steps: read, add, write. Thousands of threads did
// those steps at the same time and overwrote each other's result.
// atomicAdd() does the three steps as one indivisible operation.
//
// In real code, prefer a ready-made parallel sum (thrust::reduce, thrust::count_if,
// cupy.sum, torch.sum): it is correct and much faster than one shared counter.
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

__global__ void count_above(const float *x, int n, float threshold, int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > threshold)
        atomicAdd(count, 1);                    // FIX: one thread at a time updates *count
}

int main()
{
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);
    float *x = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) x[i] = 1.0f;

    float *d_x;
    int   *d_count;
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    count_above<<<blocks, threads>>>(d_x, n, 0.5f, d_count);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int count = 0;
    CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    printf("count = %d   (expected %d)  -> %s\n", count, n, count == n ? "PASS" : "WRONG");

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_count));
    free(x);
    return 0;
}
