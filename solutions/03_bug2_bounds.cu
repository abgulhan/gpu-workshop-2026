// 03_bug2_bounds.cu -- SOLUTION.
//
// compute-sanitizer reports two invalid READS in the kernel line that computes out[i]:
//   thread (0,0,0)   in block (0,0,0)  ->  i = 0*256 + 0     = 0    reads in[-1]
//   thread (231,0,0) in block (3,0,0)  ->  i = 3*256 + 231   = 999  reads in[1000]
// Both are outside the array. The program did not crash, it just used whatever
// happened to be in memory next to the array. Fix: treat the two ends separately.
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

__global__ void smooth(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || i == n - 1)
        out[i] = in[i];                                       // FIX: the ends have only one neighbour
    else
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
