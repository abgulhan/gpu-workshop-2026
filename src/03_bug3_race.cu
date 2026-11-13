// 03_bug3_race.cu -- BUG HUNT 3: "I get a different answer every time I run it."
//
// Goal of the program: count how many values in an array are above a threshold.
// Every element is 1 and the threshold is 0.5, so the answer must be n = 1048576.
//
// Build:  nvcc -g -lineinfo -o 03_bug3_race 03_bug3_race.cu
// Run:    ./03_bug3_race        (run it three or four times)
//
// STEP 1. Run it several times. Is the count right? Is it the same each time?
// STEP 2. No error is reported and the memory checker is silent:
//             compute-sanitizer ./03_bug3_race
//         Why? Every access is to valid memory. The problem is WHEN, not WHERE.
// STEP 3. Thousands of threads run "*count = *count + 1" at the same moment.
//         Each one reads the old value, adds one, writes it back, and they
//         overwrite each other. This is a RACE CONDITION.
//         Fix: replace the marked line with  atomicAdd(count, 1);  (read, add, write as ONE step)
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

__global__ void count_above(const float *x, int n, float threshold, int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > threshold)
        *count = *count + 1;                    // <-- the bug: all threads share *count
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
