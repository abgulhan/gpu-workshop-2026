// 04_blur_v1.cu -- blur a photo. FIRST GPU port.
//
// This is what most people write first: keep the CPU program as it is and replace
// the expensive loop by "copy to GPU, run kernel, copy back", exactly as in vector add.
//
// Build:  make -C ..                (or: nvcc -O2 -g -lineinfo -o 04_blur_v1 04_blur_v1.cu image_util.o)
// Run:    ./04_blur_v1                     (built-in test picture, 300 passes)
//         ./04_blur_v1 ../photos/earth.jpg 300       (your photo, number of passes)
// Output: out_blur_v1.png
#include "image_util.h"
#include <cstdio>
#include <cstdlib>
#include <utility>

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

// One blur pass. Each GPU thread handles ONE pixel:
// the new pixel is the average of itself and its 4 neighbours.
__global__ void blur_pass(const float *in, float *out, int w, int h)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;      // my pixel
    if (p >= w * h) return;
    int x = p % w, y = p / w;

    int left  = (x > 0)     ? p - 1 : p;                // at the border, use the pixel itself
    int right = (x < w - 1) ? p + 1 : p;
    int up    = (y > 0)     ? p - w : p;
    int down  = (y < h - 1) ? p + w : p;
    for (int c = 0; c < 3; c++)                         // red, green, blue
        out[3 * p + c] = 0.2f * (in[3 * p + c] + in[3 * left + c] + in[3 * right + c] +
                                 in[3 * up + c] + in[3 * down + c]);
}

int main(int argc, char **argv)
{
    const char *photo = (argc > 1) ? argv[1] : "-";
    int passes        = (argc > 2) ? atoi(argv[2]) : 300;
    int w, h;
    float *img = load_image(photo, &w, &h);
    size_t count = (size_t)w * h * 3, bytes = count * sizeof(float);
    float *tmp = (float *)malloc(bytes);
    printf("GPU version 1 (copy every pass): %d blur passes\n\n", passes);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    int threads = 256;
    int blocks  = (w * h + threads - 1) / threads;

    float *in = img, *out = tmp;
    double t_to_gpu = 0, t_kernel = 0, t_to_cpu = 0;
    double t0 = now_ms();
    for (int s = 0; s < passes; s++) {
        double a = now_ms();
        CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));       // CPU -> GPU
        double b = now_ms();
        blur_pass<<<blocks, threads>>>(d_in, d_out, w, h);                // one pass on the GPU
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        double c = now_ms();
        CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));     // GPU -> CPU
        double d = now_ms();
        std::swap(in, out);
        t_to_gpu += b - a; t_kernel += c - b; t_to_cpu += d - c;
    }
    double total = now_ms() - t0;

    print_preview(in, w, h, 3);
    save_image("out_blur_v1.png", in, w, h, 3);
    printf("check value (mean brightness) = %.4f\n", mean_value(in, count));
    printf("\nWhere did the time go?\n");
    printf("  copy CPU -> GPU   %9.1f ms  (%4.1f%%)\n", t_to_gpu, 100 * t_to_gpu / total);
    printf("  kernel            %9.1f ms  (%4.1f%%)\n", t_kernel, 100 * t_kernel / total);
    printf("  copy GPU -> CPU   %9.1f ms  (%4.1f%%)\n", t_to_cpu, 100 * t_to_cpu / total);
    printf("TOTAL %.1f ms   (%.3f ms per pass)\n", total, total / passes);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(img); free(tmp);
    return 0;
}

// TRY IT:
//  1. Read "Where did the time go?". Which line is the largest?
//  2. Ask the profiler the same question (it needs no change to the program):
//         nsys profile --stats=true --force-overwrite true -o blur_v1 ./04_blur_v1 ../photos/earth.jpg 100
//     Find the tables "CUDA GPU Kernel Summary" and "CUDA GPU MemOps Summary (by Time)".
//  3. The check value must equal the one printed by ./04_blur_cpu for the same photo and
//     the same number of passes. Does it?
//  4. Before you open 04_blur_v2.cu: how would you avoid the copies?
