// image_util.h -- helpers for the picture programs (plain C++, no CUDA).
//
// A picture is stored as one long array of floats:
//   3 numbers per pixel (red, green, blue), each between 0 and 255,
//   pixel (x, y) starts at index  (y * width + x) * 3
#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H
#include <cstddef>

// Read a photo (jpg, png, bmp ...). Pass "-" to get a built-in test picture.
// Photos wider than 2048 pixels are shrunk so the demo runs in a few seconds.
float *load_image(const char *path, int *width, int *height);

// Write a picture as a PNG file. channels = 3 for colour, 1 for grayscale.
void save_image(const char *path, const float *img, int width, int height, int channels);

// Print a small preview of the picture in the terminal (brighter = denser character).
void print_preview(const float *img, int width, int height, int channels);

// Average of all the numbers: a quick way to check that two versions agree.
double mean_value(const float *a, size_t count);

// A clock, in milliseconds.
double now_ms();

#endif
