// 04_blur_v2.cu -- blur a photo. GPU port with the picture kept on the GPU.
//
// The profile of version 1 showed where the time went: copying the picture between
// CPU and GPU on every pass. Here the picture is copied to the GPU once, all passes
// run there (the two GPU arrays just swap roles), and the result is copied back once.
//
// Build:  make -C ..                (or: nvcc -O2 -g -lineinfo -o 04_blur_v2 04_blur_v2.cu image_util.o)
// Run:    ./04_blur_v2                       (built-in test picture, 300 passes)
//         ./04_blur_v2 ../photos/earth.jpg 300         (your photo, number of passes)
//         ./04_blur_v2 ../photos/earth.jpg 300 32      (third value: threads per block, default 256)
// Output: out_blur_v2.png
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

// One blur pass. Each GPU thread handles ONE pixel. (Identical to version 1.)
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
    int threads       = (argc > 3) ? atoi(argv[3]) : 256;     // threads per block
    int w, h;
    float *img = load_image(photo, &w, &h);
    size_t count = (size_t)w * h * 3, bytes = count * sizeof(float);
    printf("GPU version 2 (picture stays on the GPU): %d blur passes, %d threads per block\n\n",
           passes, threads);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    int blocks = (w * h + threads - 1) / threads;

    double t0 = now_ms();
    CHECK(cudaMemcpy(d_in, img, bytes, cudaMemcpyHostToDevice));          // CPU -> GPU, once
    double t1 = now_ms();
    for (int s = 0; s < passes; s++) {
        blur_pass<<<blocks, threads>>>(d_in, d_out, w, h);
        CHECK(cudaGetLastError());
        std::swap(d_in, d_out);           // the result becomes the next input: only two
                                          // pointers swap, no picture data moves
    }
    CHECK(cudaDeviceSynchronize());       // wait for all the passes to finish
    double t2 = now_ms();
    CHECK(cudaMemcpy(img, d_in, bytes, cudaMemcpyDeviceToHost));          // GPU -> CPU, once
    double t3 = now_ms();
    double total = t3 - t0;

    print_preview(img, w, h, 3);
    save_image("out_blur_v2.png", img, w, h, 3);
    printf("check value (mean brightness) = %.4f\n", mean_value(img, count));
    printf("\nWhere did the time go?\n");
    printf("  copy CPU -> GPU   %9.1f ms  (%4.1f%%)\n", t1 - t0, 100 * (t1 - t0) / total);
    printf("  kernel            %9.1f ms  (%4.1f%%)\n", t2 - t1, 100 * (t2 - t1) / total);
    printf("  copy GPU -> CPU   %9.1f ms  (%4.1f%%)\n", t3 - t2, 100 * (t3 - t2) / total);
    printf("TOTAL %.1f ms   (%.3f ms per pass)\n", total, total / passes);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(img);
    return 0;
}

// TRY IT:
//  1. Compare TOTAL and the check value with ./04_blur_v1. Same picture, how much faster?
//  2. Threads per block (third value): try 1, 32, 256 and 1024.
//         ./04_blur_v2 ../photos/earth.jpg 300 1
//     Which is slowest? One thread per block is a classic beginner mistake.
//  3. Delete the std::swap line, rebuild and run. The picture is not blurred at all. Why?
//     (Which array does every pass read, and which one is copied back at the end?)
