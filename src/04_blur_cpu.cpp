// 04_blur_cpu.cpp -- blur a photo. CPU version (the baseline).
//
// One blur pass: every pixel becomes the average of itself and its 4 neighbours.
// One pass is barely visible, so we do 300 passes. More passes = more blur.
//
// Build:  make -C ..                (or: g++ -O2 -fopenmp -o 04_blur_cpu 04_blur_cpu.cpp image_util.o)
// Run:    OMP_NUM_THREADS=1 ./04_blur_cpu                  (one CPU core, test picture)
//         OMP_NUM_THREADS=4 ./04_blur_cpu ../photos/earth.jpg        (four CPU cores, your photo)
//         ./04_blur_cpu ../photos/earth.jpg 300                      (number of passes)
// Output: out_blur_cpu.png
#include "image_util.h"
#include <cstdio>
#include <cstdlib>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

void blur_pass(const float *in, float *out, int w, int h)
{
    // OpenMP: share the rows of this loop among the CPU cores.
    // OMP_NUM_THREADS=4 in front of the command means "use 4 cores".
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int p     = y * w + x;                      // this pixel
            int left  = (x > 0)     ? p - 1 : p;        // at the border, use the pixel itself
            int right = (x < w - 1) ? p + 1 : p;
            int up    = (y > 0)     ? p - w : p;
            int down  = (y < h - 1) ? p + w : p;
            for (int c = 0; c < 3; c++)                 // red, green, blue
                out[3 * p + c] = 0.2f * (in[3 * p + c] + in[3 * left + c] + in[3 * right + c] +
                                         in[3 * up + c] + in[3 * down + c]);
        }
    }
}

int main(int argc, char **argv)
{
    const char *photo = (argc > 1) ? argv[1] : "-";
    int passes        = (argc > 2) ? atoi(argv[2]) : 300;
    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif
    int w, h;
    float *img = load_image(photo, &w, &h);
    size_t count = (size_t)w * h * 3;
    float *tmp = (float *)malloc(count * sizeof(float));
    printf("CPU version: %d blur passes, %d CPU core(s)\n\n", passes, cores);

    float *in = img, *out = tmp;
    double t0 = now_ms();
    for (int s = 0; s < passes; s++) {
        blur_pass(in, out, w, h);
        std::swap(in, out);               // the result becomes the input of the next pass
    }
    double total = now_ms() - t0;

    print_preview(in, w, h, 3);
    save_image("out_blur_cpu.png", in, w, h, 3);
    printf("check value (mean brightness) = %.4f\n", mean_value(in, count));
    printf("\nTOTAL %.1f ms   (%.3f ms per pass)\n", total, total / passes);

    free(img); free(tmp);
    return 0;
}

// TRY IT:
//  1. Run with 1, 2 and 4 cores (OMP_NUM_THREADS=...; a Roar session from the handout has
//     4 cores). Does the time halve each time?
//  2. Run with 30 passes, keep the picture:   mv out_blur_cpu.png blur_30.png
//     Then run with 3000 passes and open blur_30.png and out_blur_cpu.png side by side.
