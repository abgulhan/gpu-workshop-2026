// image_util.cpp -- reading, writing and previewing pictures (plain C++, no CUDA).
// Uses the public-domain stb_image headers (https://github.com/nothings/stb).
#include "image_util.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// A built-in picture with sharp edges, flat areas and smooth gradients.
static void test_pattern(unsigned char *pix, int W, int H)
{
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float r = 255.0f * x / W, g = 255.0f * y / H, b = 140.0f;          // colour gradient
            if (x < W / 3) {                                                   // checkerboard
                float v = (((x / 60) + (y / 60)) % 2) ? 235.0f : 25.0f;
                r = g = b = v;
            }
            float dx = x - 0.62f * W, dy = y - 0.55f * H, rad = 0.28f * H;
            if (dx * dx + dy * dy < rad * rad) { r = 250; g = 205; b = 40; }   // yellow disc
            dx = x - 0.85f * W; dy = y - 0.25f * H; rad = 0.13f * H;
            if (dx * dx + dy * dy < rad * rad) { r = 255; g = 255; b = 255; }  // white disc
            if (y > 0.88f * H && ((x + y) / 12) % 2) { r = 20; g = 20; b = 60; }   // thin stripes
            unsigned char *p = pix + ((size_t)y * W + x) * 3;
            p[0] = (unsigned char)r; p[1] = (unsigned char)g; p[2] = (unsigned char)b;
        }
    }
}

// Phone cameras often store a photo sideways and add a note saying "turn me when you
// show me" (the EXIF orientation, a number from 1 to 8). This reads that number
// from a JPEG file. 1 means "already upright"; that is also the answer if there is no note.
static int jpeg_orientation(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    static unsigned char d[1 << 17];
    size_t n = fread(d, 1, sizeof(d), f);
    fclose(f);
    if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return 1;                // not a JPEG file
    size_t pos = 2;
    while (pos + 4 <= n && d[pos] == 0xFF) {
        int marker = d[pos + 1];
        size_t len = ((size_t)d[pos + 2] << 8) | d[pos + 3];
        if (marker == 0xDA) break;                                      // picture data starts
        if (marker == 0xE1 && pos + 4 + len <= n + 2 && len >= 16 &&
            memcmp(d + pos + 4, "Exif\0\0", 6) == 0) {
            const unsigned char *t = d + pos + 10;                      // the note itself
            size_t tlen = len - 8;
            bool little = (t[0] == 'I');
            #define U16(o) (little ? (t[o] | (t[(o) + 1] << 8)) : ((t[o] << 8) | t[(o) + 1]))
            #define U32(o) (little ? (U16(o) | ((size_t)U16((o) + 2) << 16)) : (((size_t)U16(o) << 16) | U16((o) + 2)))
            size_t ifd = U32(4);
            if (ifd + 2 > tlen) return 1;
            int entries = U16(ifd);
            for (int e = 0; e < entries; e++) {
                size_t o = ifd + 2 + 12 * (size_t)e;
                if (o + 12 > tlen) break;
                if (U16(o) == 0x0112) {                                 // the orientation entry
                    int v = U16(o + 8);
                    return (v >= 1 && v <= 8) ? v : 1;
                }
            }
            #undef U16
            #undef U32
            return 1;
        }
        pos += 2 + len;
    }
    return 1;
}

// Turn or mirror the picture so that it is upright. May swap width and height.
static unsigned char *make_upright(unsigned char *pix, int *W, int *H, int orientation)
{
    if (orientation <= 1) return pix;
    int w = *W, h = *H;
    bool swap = (orientation >= 5);                                     // 5 to 8: quarter turns
    int ow = swap ? h : w, oh = swap ? w : h;
    unsigned char *out = (unsigned char *)malloc((size_t)ow * oh * 3);
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int sx = x, sy = y;                                         // where this pixel comes from
            switch (orientation) {
                case 2: sx = w - 1 - x; sy = y;         break;          // mirrored left-right
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;          // upside down
                case 4: sx = x;         sy = h - 1 - y; break;          // mirrored top-bottom
                case 5: sx = y;         sy = x;         break;
                case 6: sx = y;         sy = h - 1 - x; break;          // quarter turn clockwise
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;
                case 8: sx = w - 1 - y; sy = x;         break;          // quarter turn anticlockwise
            }
            memcpy(out + ((size_t)y * ow + x) * 3, pix + ((size_t)sy * w + sx) * 3, 3);
        }
    }
    stbi_image_free(pix);
    *W = ow; *H = oh;
    return out;
}

float *load_image(const char *path, int *width, int *height)
{
    int W = 0, H = 0;
    unsigned char *pix = NULL;
    bool from_file = (path != NULL && strcmp(path, "-") != 0);

    if (from_file) {
        int channels_in_file;
        pix = stbi_load(path, &W, &H, &channels_in_file, 3);       // always give us RGB
        if (!pix) {
            fprintf(stderr, "Cannot read picture '%s': %s\n", path, stbi_failure_reason());
            exit(1);
        }
        pix = make_upright(pix, &W, &H, jpeg_orientation(path));    // phone photos: see above
    } else {
        W = 1920; H = 1080;
        pix = (unsigned char *)malloc((size_t)W * H * 3);
        test_pattern(pix, W, H);
    }

    // Shrink big photos by a whole-number factor k (average each k x k group of pixels).
    int k = 1;
    while (W / k > 2048) k++;
    int w = W / k, h = H / k;

    float *img = (float *)malloc((size_t)w * h * 3 * sizeof(float));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0;
                for (int yy = 0; yy < k; yy++)
                    for (int xx = 0; xx < k; xx++)
                        sum += pix[((size_t)(y * k + yy) * W + (x * k + xx)) * 3 + c];
                img[((size_t)y * w + x) * 3 + c] = sum / (float)(k * k);
            }
        }
    }

    if (from_file) stbi_image_free(pix); else free(pix);
    printf("picture: %s, %d x %d pixels", from_file ? path : "built-in test picture", w, h);
    if (k > 1) printf(" (shrunk from %d x %d)", W, H);
    printf("\n");

    *width = w; *height = h;
    return img;
}

void save_image(const char *path, const float *img, int width, int height, int channels)
{
    size_t count = (size_t)width * height * channels;
    unsigned char *bytes = (unsigned char *)malloc(count);
    for (size_t i = 0; i < count; i++) {
        float v = img[i];
        bytes[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v + 0.5f));
    }
    if (stbi_write_png(path, width, height, channels, bytes, width * channels))
        printf("wrote %s\n", path);
    else
        fprintf(stderr, "could not write %s\n", path);
    free(bytes);
}

void print_preview(const float *img, int width, int height, int channels)
{
    if (!getenv("SHOW_PREVIEW")) return;       // off by default; SHOW_PREVIEW=1 ./program turns it on
    const char *shades = " .:-=+*#%@";
    const int cols = 72, rows = 22;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int x = (int)((c + 0.5) * width / cols), y = (int)((r + 0.5) * height / rows);
            const float *p = img + ((size_t)y * width + x) * channels;
            float v = 0;
            for (int ch = 0; ch < channels; ch++) v += p[ch];
            v /= (float)channels;
            int level = (int)(v / 255.0f * 9.999f);
            if (level < 0) level = 0;
            if (level > 9) level = 9;
            putchar(shades[level]);
        }
        putchar('\n');
    }
}

double mean_value(const float *a, size_t count)
{
    double sum = 0;
    for (size_t i = 0; i < count; i++) sum += a[i];
    return sum / (double)count;
}
