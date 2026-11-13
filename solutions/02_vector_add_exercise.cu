// 02_vector_add_exercise.cu -- SOLUTION.
//
// Build:  make -C .. solutions/02_vector_add_exercise   (run from the folder of this file)
// Run:    ./02_vector_add_exercise
// Expected output:  c[0] = 3, c[n-1] = 3  -> PASS
#include <cstdio>
#include <cstdlib>

__global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }

    // STEP 1: allocate GPU memory ("d_" = device = on the GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // STEP 2: copy in (host = CPU, device = GPU)
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // STEP 3: launch the kernel
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // STEP 4: copy out. cudaMemcpy first waits for the kernel to finish,
    // so no cudaDeviceSynchronize() is needed here.
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("c[0] = %g, c[n-1] = %g  -> %s\n", c[0], c[n - 1],
           (c[0] == 3.0f && c[n - 1] == 3.0f) ? "PASS" : "FAIL");

    // STEP 5: free
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}
