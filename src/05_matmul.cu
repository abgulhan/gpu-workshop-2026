// 05_matmul.cu -- matrix multiplication C = A x B with a hand-written kernel.
//
// This is the "obvious" GPU version: one thread per element of C, each thread
// does one row-times-column sum. Compare its time with the PyTorch one-liner in
// ../python/05_matmul_torch.py, which calls NVIDIA's tuned library.
//
// Build:  make -C ..                (or: nvcc -O2 -o 05_matmul 05_matmul.cu)
// Run:    ./05_matmul               (4096 x 4096)
//         ./05_matmul 2048          (matrix size n)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

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

// Each thread computes ONE element of C: row "row" of A times column "col" of B.
// Matrices are stored row after row in one long array: element (row, col) is at row * n + col.
// This version is correct but slow: every thread walks through n values of A and n of B,
// and the values of B are far apart in memory. Tuned libraries avoid exactly that.
__global__ void matmul(const float *a, const float *b, float *c, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n * n) return;
    int row = p / n, col = p % n;

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += a[row * n + k] * b[k * n + col];
    c[p] = sum;
}

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 4096;
    size_t bytes = (size_t)n * n * sizeof(float);
    printf("hand-written kernel: %d x %d matrices\n", n, n);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (size_t i = 0; i < (size_t)n * n; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n * n + threads - 1) / threads;

    matmul<<<1, 1>>>(d_a, d_b, d_c, 1);             // warm-up launch (not timed)
    CHECK(cudaDeviceSynchronize());

    double t0 = now_ms();
    matmul<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());                 // wait, so the timing is real
    double t_kernel = now_ms() - t0;

    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Check three elements against the CPU.
    int rows[3] = {0, n / 2, n - 1}, cols[3] = {0, n / 3, n - 1};
    double worst = 0;
    for (int t = 0; t < 3; t++) {
        double sum = 0;
        for (int k = 0; k < n; k++) sum += (double)a[(size_t)rows[t] * n + k] * b[(size_t)k * n + cols[t]];
        double rel = fabs(sum - c[(size_t)rows[t] * n + cols[t]]) / sum;
        if (rel > worst) worst = rel;
    }
    printf("largest relative difference from the CPU on 3 test elements: %.1e  -> %s\n",
           worst, worst < 1e-4 ? "results match" : "MISMATCH");
    printf("GPU kernel (hand-written)  %9.1f ms\n", t_kernel);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(a); free(b); free(c);
    return 0;
}

// TRY IT:
//  1. Run ./05_matmul 1024, 2048 and 4096. The matrix has 4 times more elements each
//     time. By how much does the time grow? (Each element also needs twice the work.)
//  2. Run python3 ../python/05_matmul_torch.py and compare "GPU multiply" with the
//     time printed here.
