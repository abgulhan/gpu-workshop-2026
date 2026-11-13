// 04_image_filters.cu -- two photo filters on the GPU: grayscale and edge detection.
//
// A photo is the ideal GPU problem: millions of pixels, and every pixel can be
// worked out on its own. ONE GPU THREAD PER PIXEL.
//
// Build:  make -C ..                (or: nvcc -O2 -g -lineinfo -o 04_image_filters 04_image_filters.cu image_util.o)
// Run:    ./04_image_filters                 (built-in test picture)
//         ./04_image_filters ../photos/earth.jpg       (your own photo)
// Output: out_original.png, out_gray.png, out_edges.png   (open them side by side)
#include "image_util.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

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

// Filter 1: colour -> grayscale. Each thread reads the 3 colour values of ITS pixel
// and writes one brightness value. Same shape as vector add.
__global__ void to_gray(const float *rgb, float *gray, int n_pixels)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;      // my pixel
    if (p >= n_pixels) return;
    gray[p] = 0.299f * rgb[3 * p] + 0.587f * rgb[3 * p + 1] + 0.114f * rgb[3 * p + 2];
}

// Filter 2: edge detection. Each thread compares the pixels on the left of its pixel
// with those on the right (gx), and those above with those below (gy).
// A big difference means an edge.
__global__ void find_edges(const float *gray, float *edges, int w, int h)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= w * h) return;
    int x = p % w, y = p / w;

    // Pixels on the border have no neighbours outside the picture (remember bug 2).
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) { edges[p] = 0.0f; return; }

    float gx = -gray[p - w - 1] + gray[p - w + 1]
               - 2.0f * gray[p - 1] + 2.0f * gray[p + 1]
               - gray[p + w - 1] + gray[p + w + 1];
    float gy = -gray[p - w - 1] - 2.0f * gray[p - w] - gray[p - w + 1]
               + gray[p + w - 1] + 2.0f * gray[p + w] + gray[p + w + 1];
    float strength = sqrtf(gx * gx + gy * gy);
    edges[p] = strength > 255.0f ? 255.0f : strength;
}

int main(int argc, char **argv)
{
    const char *photo = (argc > 1) ? argv[1] : "-";
    int w, h;
    float *rgb = load_image(photo, &w, &h);
    int n_pixels = w * h;
    float *gray  = (float *)malloc((size_t)n_pixels * sizeof(float));
    float *edges = (float *)malloc((size_t)n_pixels * sizeof(float));

    float *d_rgb, *d_gray, *d_edges;
    CHECK(cudaMalloc(&d_rgb,   (size_t)n_pixels * 3 * sizeof(float)));
    CHECK(cudaMalloc(&d_gray,  (size_t)n_pixels * sizeof(float)));
    CHECK(cudaMalloc(&d_edges, (size_t)n_pixels * sizeof(float)));

    int threads = 256;
    int blocks  = (n_pixels + threads - 1) / threads;
    printf("%d pixels -> %d blocks of %d threads\n\n", n_pixels, blocks, threads);

    to_gray<<<1, 1>>>(d_rgb, d_gray, 0);        // warm-up launches (the first launch of a
    find_edges<<<1, 1>>>(d_gray, d_edges, 0, 0);    // kernel is slow), so the timing below is fair
    CHECK(cudaDeviceSynchronize());

    double t0 = now_ms();
    CHECK(cudaMemcpy(d_rgb, rgb, (size_t)n_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    double t1 = now_ms();

    to_gray<<<blocks, threads>>>(d_rgb, d_gray, n_pixels);
    CHECK(cudaGetLastError());
    find_edges<<<blocks, threads>>>(d_gray, d_edges, w, h);    // uses the result of the first kernel,
    CHECK(cudaGetLastError());                                 // which never left the GPU
    CHECK(cudaDeviceSynchronize());
    double t2 = now_ms();

    CHECK(cudaMemcpy(gray,  d_gray,  (size_t)n_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(edges, d_edges, (size_t)n_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    double t3 = now_ms();

    if (getenv("SHOW_PREVIEW")) printf("Edges, as seen in the terminal:\n");
    print_preview(edges, w, h, 1);
    printf("\n");
    save_image("out_original.png", rgb, w, h, 3);
    save_image("out_gray.png", gray, w, h, 1);
    save_image("out_edges.png", edges, w, h, 1);

    printf("\ncopy CPU -> GPU   %8.2f ms\n", t1 - t0);
    printf("two kernels       %8.2f ms\n", t2 - t1);
    printf("copy GPU -> CPU   %8.2f ms\n", t3 - t2);
    printf("check value (mean brightness of the edges picture) = %.4f\n", mean_value(edges, n_pixels));

    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));
    CHECK(cudaFree(d_edges));
    free(rgb); free(gray); free(edges);
    return 0;
}

// TRY IT:
//  1. Make a "negative" of the photo: in to_gray, write 255 minus the brightness.
//  2. Show only strong edges: in find_edges, write 255 if strength > 80, else 0.
//  3. Remove the border test in find_edges, rebuild, and run
//         compute-sanitizer ./04_image_filters
//     What does it report?
