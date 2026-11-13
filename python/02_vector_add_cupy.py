"""02_vector_add_cupy.py -- the SAME kernel as src/02_vector_add.cu, launched from Python.

CuPy is "NumPy on the GPU", and it can also run a CUDA kernel that you give it as text.
The kernel below is CUDA C++, copied from src/02_vector_add.cu. The only addition is the
line  extern "C"  (it tells the compiler to keep the plain name "add" so Python can find it).

Everything AROUND the kernel gets shorter. Compare with the .cu file:

    C++ (src/02_vector_add.cu)                   Python (this file)
    -------------------------------------------  ------------------------------------
    cudaMalloc + cudaMemcpy(... HostToDevice)    a_gpu = cp.asarray(a)
    add<<<blocks, threads>>>(d_a, d_b, d_c, n)   add((blocks,), (threads,), (a_gpu, b_gpu, c_gpu, n))
    cudaDeviceSynchronize()                      cp.cuda.Stream.null.synchronize()
    cudaMemcpy(... DeviceToHost)                 c = cp.asnumpy(c_gpu)
    cudaFree                                     (automatic)

Run:    python3 02_vector_add_cupy.py            (default: 67 million elements)
        python3 02_vector_add_cupy.py 1000000    (choose the number of elements)
"""
import sys
import time

import numpy as np
import cupy as cp

# ---------------------------------------------------------------- GPU kernel (CUDA C++)
kernel_source = r'''
extern "C"
__global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which element is mine?
    if (i < n)                                      // the last block may have spare threads
        c[i] = a[i] + b[i];
}
'''
add = cp.RawKernel(kernel_source, "add")    # compiled for the GPU the first time it is launched


def wait():
    """Wait until the GPU has finished (cudaDeviceSynchronize in the C++ version)."""
    cp.cuda.Stream.null.synchronize()


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000


n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 26
print(f"n = {n} elements ({n * 4 / 1e6:.0f} MB per array)\n")

# Arrays in CPU memory
a = np.full(n, 1.0, dtype=np.float32)
b = np.full(n, 2.0, dtype=np.float32)

# ---------------------------------------------------------------- CPU (NumPy)
c_cpu = np.full(n, 0.0, dtype=np.float32)                # made before the clock starts, as in the C++ program
t0 = time.perf_counter()
np.add(a, b, out=c_cpu)                                 # c_cpu = a + b, written into the existing array
t_cpu = ms_since(t0)

# ---------------------------------------------------------------- GPU
t0 = time.perf_counter()
cp.zeros(1)                                             # wakes up the GPU
tiny = cp.zeros(1, cp.float32)
add((1,), (1,), (tiny, tiny, tiny, np.int32(0)))        # warm-up launch: compiles our kernel
_ = tiny + tiny                                         # warm-up for CuPy's own addition
wait()
t_start = ms_since(t0)

t0 = time.perf_counter()
a_gpu = cp.asarray(a)                                   # allocate on the GPU and copy CPU -> GPU
b_gpu = cp.asarray(b)
c_gpu = cp.empty(n, dtype=cp.float32)                   # allocate only
wait()
t_in = ms_since(t0)

threads = 256                                           # threads per block
blocks = (n + threads - 1) // threads                   # enough blocks to cover n elements

# The launch. (blocks,) and (threads,) are written as one-element tuples because 2-D and 3-D
# layouts also exist. np.int32(n): the kernel expects a 32-bit "int", and a plain Python number
# would be sent as 64 bits.

t0 = time.perf_counter()
add((blocks,), (threads,), (a_gpu, b_gpu, c_gpu, np.int32(n)))     # add<<<blocks, threads>>>(...)
wait()
t_kernel = ms_since(t0)

t0 = time.perf_counter()
c = cp.asnumpy(c_gpu)                                   # copy GPU -> CPU
t_out = ms_since(t0)

# The library way: no kernel at all. CuPy arrays behave like NumPy arrays.
t0 = time.perf_counter()
c_lib = a_gpu + b_gpu
wait()
t_lib = ms_since(t0)

# ---------------------------------------------------------------- report
diff = float(np.abs(c - c_cpu).max())
print(f"CPU (NumPy, one core)        {t_cpu:9.2f} ms")
print(f"GPU start-up (one-time)      {t_start:9.2f} ms")
print(f"GPU copy in                  {t_in:9.2f} ms")
print(f"GPU kernel (ours)            {t_kernel:9.2f} ms")
print(f"GPU copy out                 {t_out:9.2f} ms")
print(f"GPU, CuPy's own a + b        {t_lib:9.2f} ms   (no kernel written)")
print()
print(f"c[0] = {c[0]:g}, c[n-1] = {c[-1]:g}")
print(f"Largest difference CPU vs GPU: {diff:g}  -> {'results match' if diff == 0 else 'MISMATCH'}")
print(f"Speedup, kernel only:        {t_cpu / t_kernel:6.1f}x")
print(f"Speedup, including copies:   {t_cpu / (t_in + t_kernel + t_out):6.1f}x")

# TRY IT:
#  1. Comment out the wait() right after the launch (the line under "add((blocks,), ...").
#     What happens to "GPU kernel (ours)"? Where did that time go?
#  2. Change threads to 1, then to 1024. How does "GPU kernel (ours)" change?
#  3. Change the kernel to compute c[i] = a[i] * b[i] + 10 and just run again (there is no
#     build step). c[0] becomes 12, and the check now says MISMATCH, because the CPU line
#     still computes a + b. Change the CPU line too.
