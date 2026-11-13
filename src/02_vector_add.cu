// 02_vector_add.cu -- c[i] = a[i] + b[i] on the CPU and on the GPU, with timing.
//
// Every GPU program has the same five steps. Look for them in main():
//   1 allocate GPU memory   2 copy in   3 launch the kernel   4 copy out   5 free
//
// Build:  make -C ..                (or: nvcc -O2 -o 02_vector_add 02_vector_add.cu)
// Run:    ./02_vector_add            (default: 67 million elements)
//         ./02_vector_add 1000000    (choose the number of elements)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------- GPU kernel
// Runs once per GPU thread. Each thread handles ONE element.
__global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which element is mine?
    if (i < n)                                      // the last block may have spare threads
        c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- CPU version
void add_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1 << 26;
    size_t bytes = (size_t)n * sizeof(float);
    printf("n = %d elements (%.0f MB per array)\n\n", n, bytes / 1e6);

    // Host (CPU) arrays
    float *a     = (float *)malloc(bytes);
    float *b     = (float *)malloc(bytes);
    float *c_cpu = (float *)malloc(bytes);
    float *c_gpu = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f * (i % 1000); b[i] = 2.0f; c_cpu[i] = 0; c_gpu[i] = 0; }

    // ------------------------------------------------------------ CPU
    double t0 = now_ms();
    add_cpu(a, b, c_cpu, n);
    double t_cpu = now_ms() - t0;

    // ------------------------------------------------------------ GPU
    // STEP 1: allocate GPU memory. The first CUDA call also starts up the GPU,
    // a one-time cost that we time separately as "start-up".
    t0 = now_ms();
    float *d_a, *d_b, *d_c;                 // "d_" = device = arrays in GPU memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    add<<<1, 1>>>(d_a, d_b, d_c, 0);        // warm-up launch: the first launch of a kernel is slow
    cudaDeviceSynchronize();
    double t_init = now_ms() - t0;

    // STEP 2: copy in. The inputs go from CPU memory to GPU memory.
    t0 = now_ms();
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    double t_h2d = now_ms() - t0;

    // STEP 3: launch the kernel on enough threads to cover all n elements.
    int threads = 256;                              // threads per block
    int blocks  = (n + threads - 1) / threads;      // enough blocks to cover n (round up)
    t0 = now_ms();
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();                        // wait, so the timing is real
    double t_kernel = now_ms() - t0;

    // STEP 4: copy out. The result comes back to CPU memory.
    t0 = now_ms();
    cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    double t_d2h = now_ms() - t0;

    // ------------------------------------------------------------ check + report
    double max_err = 0;
    for (int i = 0; i < n; i++) max_err = fmax(max_err, fabs((double)c_cpu[i] - c_gpu[i]));

    double t_gpu = t_h2d + t_kernel + t_d2h;
    printf("CPU loop (1 core)          %9.2f ms\n\n", t_cpu);
    printf("GPU start-up (one-time)    %9.2f ms\n", t_init);
    printf("GPU copy CPU -> GPU        %9.2f ms\n", t_h2d);
    printf("GPU kernel                 %9.2f ms\n", t_kernel);
    printf("GPU copy GPU -> CPU        %9.2f ms\n", t_d2h);
    printf("GPU copy + kernel + copy   %9.2f ms\n\n", t_gpu);
    printf("Largest difference CPU vs GPU: %g  -> %s\n", max_err, max_err == 0 ? "results match" : "MISMATCH");
    printf("Speedup, kernel only:        %6.1fx\n", t_cpu / t_kernel);
    printf("Speedup, including copies:   %6.1fx\n", t_cpu / t_gpu);

    // STEP 5: free the GPU memory (and the CPU memory).
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c_cpu); free(c_gpu);
    return 0;
}

// TRY IT:
//  1. Leave the one-time start-up aside. Which of the three GPU stages (copy in, kernel,
//     copy out) is the largest? What does that tell you?
//  2. Run with a small problem:  ./02_vector_add 1000   Is the GPU still worth it?
//  3. Comment out the cudaDeviceSynchronize() in STEP 3 (the one marked "wait, so the
//     timing is real") and rebuild. The "GPU kernel" time drops to almost zero.
//     Did the GPU really get faster? Where did the time go? (Look at "copy GPU -> CPU".)
