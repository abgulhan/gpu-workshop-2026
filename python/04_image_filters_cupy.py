"""04_image_filters_cupy.py -- the SAME two kernels as src/04_image_filters.cu, from Python.

ONE GPU THREAD PER PIXEL. The two kernels are CUDA C++, copied from the .cu file (plus the
line  extern "C"  so Python can find them by name). Loading and saving the picture, which
needed a helper file in C++, is one line each here.

Run:    python3 04_image_filters_cupy.py ../photos/earth.jpg     (or your own jpg / png)
        python3 04_image_filters_cupy.py                         (no name: photos/earth.jpg)
Output: out_gray_py.png, out_edges_py.png   (in the folder you run it from)
"""
import os
import sys
import time

import numpy as np
import cupy as cp
from PIL import Image, ImageOps

# ---------------------------------------------------------------- GPU kernels (CUDA C++)
kernel_source = r'''
// Filter 1: colour -> grayscale. Each thread reads the 3 colour values of ITS pixel
// and writes one brightness value. Same shape as vector add.
extern "C"
__global__ void to_gray(const float *rgb, float *gray, int n_pixels)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;      // my pixel
    if (p >= n_pixels) return;
    gray[p] = 0.299f * rgb[3 * p] + 0.587f * rgb[3 * p + 1] + 0.114f * rgb[3 * p + 2];
}

// Filter 2: edge detection. Each thread compares the pixels on the left of its pixel
// with those on the right (gx), and those above with those below (gy).
// A big difference means an edge.
extern "C"
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
'''
module = cp.RawModule(code=kernel_source)       # one piece of CUDA text, two kernels
to_gray = module.get_function("to_gray")
find_edges = module.get_function("find_edges")


def wait():
    """Wait until the GPU has finished (cudaDeviceSynchronize in the C++ version)."""
    cp.cuda.Stream.null.synchronize()


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- load the photo (CPU)
here = os.path.dirname(os.path.abspath(__file__))
photo = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "photos", "earth.jpg")
picture = Image.open(photo)
picture = ImageOps.exif_transpose(picture).convert("RGB")   # phone photos: turn upright
factor = 1                                               # keep large photos manageable:
while picture.width // factor > 2048:                    # shrink by a whole number, with the
    factor += 1                                          # same rule as the C++ programs
if factor > 1:
    picture = picture.crop((0, 0, picture.width // factor * factor, picture.height // factor * factor))
    picture = picture.reduce(factor)
w, h = picture.size
rgb = np.asarray(picture, dtype=np.float32)              # h x w x 3 numbers, 0 to 255
n_pixels = w * h

threads = 256
blocks = (n_pixels + threads - 1) // threads
print(f"{photo}: {w} x {h} = {n_pixels} pixels -> {blocks} blocks of {threads} threads\n")

# warm-up launches (the first launch compiles the kernels), so the timing below is fair
tiny = cp.zeros(3, cp.float32)
to_gray((1,), (1,), (tiny, tiny, np.int32(0)))
find_edges((1,), (1,), (tiny, tiny, np.int32(0), np.int32(0)))
wait()

# ---------------------------------------------------------------- GPU
t0 = time.perf_counter()
rgb_gpu = cp.asarray(rgb)                                # copy CPU -> GPU
gray_gpu = cp.empty(n_pixels, dtype=cp.float32)
edges_gpu = cp.empty(n_pixels, dtype=cp.float32)
wait()
t_in = ms_since(t0)

t0 = time.perf_counter()
to_gray((blocks,), (threads,), (rgb_gpu, gray_gpu, np.int32(n_pixels)))
find_edges((blocks,), (threads,), (gray_gpu, edges_gpu, np.int32(w), np.int32(h)))   # gray never left the GPU
wait()
t_kernels = ms_since(t0)

t0 = time.perf_counter()
gray = cp.asnumpy(gray_gpu)                              # copy GPU -> CPU
edges = cp.asnumpy(edges_gpu)
t_out = ms_since(t0)

# ---------------------------------------------------------------- save the pictures (CPU)
Image.fromarray(gray.reshape(h, w).astype(np.uint8)).save("out_gray_py.png")
Image.fromarray(edges.reshape(h, w).astype(np.uint8)).save("out_edges_py.png")
print("wrote out_gray_py.png")
print("wrote out_edges_py.png\n")

print(f"copy CPU -> GPU   {t_in:9.2f} ms")
print(f"two kernels       {t_kernels:9.2f} ms")
print(f"copy GPU -> CPU   {t_out:9.2f} ms")
print(f"check value (mean brightness of the edges picture) = {edges.mean():.4f}")
# ./04_image_filters prints the same check value for the same photo. The two agree to about two
# decimals, not exactly: the programs round the shrunken photo slightly differently.

# TRY IT:
#  1. Negative: in to_gray, write 255 minus the brightness. Run again, open out_gray_py.png.
#  2. Threshold: in find_edges, write 255 if strength > 80, else 0. Sharper lines?
#  3. The library way: gray2 = rgb_gpu @ cp.asarray([0.299, 0.587, 0.114], cp.float32)
#     gives the grayscale picture with no kernel at all.
