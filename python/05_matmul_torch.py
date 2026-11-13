"""05_matmul_torch.py -- matrix multiplication C = A x B with PyTorch.

No kernel, no blocks, no threads: the library does it. The same three rules
still apply: copy the data to the GPU, wait before you stop the clock, and
check the answer.

Run:    python3 05_matmul_torch.py          (4096 x 4096)
        python3 05_matmul_torch.py 2048     (matrix size n)
"""
import sys
import time

import torch

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
print(f"PyTorch {torch.__version__}: {n} x {n} matrices")

a = torch.rand(n, n)            # created in CPU memory
b = torch.rand(n, n)

# ---------------------------------------------------------------- CPU
t0 = time.perf_counter()
c_cpu = a @ b
t_cpu = (time.perf_counter() - t0) * 1000
print(f"CPU ({torch.get_num_threads()} cores)        {t_cpu:9.1f} ms")

if not torch.cuda.is_available():
    sys.exit("No GPU visible to PyTorch. Run nvidia-smi to check.")

# ---------------------------------------------------------------- GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")

t0 = time.perf_counter()
a_gpu = a.to("cuda")            # copy CPU -> GPU   (cudaMalloc + cudaMemcpy in Part 2)
b_gpu = b.to("cuda")
torch.cuda.synchronize()
t_copy = (time.perf_counter() - t0) * 1000

_ = a_gpu[:64, :64] @ b_gpu[:64, :64]   # warm-up: the first call loads the library
torch.cuda.synchronize()

t0 = time.perf_counter()
c_gpu = a_gpu @ b_gpu           # THE WHOLE GPU PROGRAM IS THIS LINE
torch.cuda.synchronize()        # wait, so the timing is real (cudaDeviceSynchronize in Part 2)
t_gpu = (time.perf_counter() - t0) * 1000

diff = ((c_cpu - c_gpu.cpu()).abs().max() / c_cpu.abs().max()).item()   # .cpu() copies GPU -> CPU

print(f"start-up + copy to GPU {t_copy:9.1f} ms   (start-up is one-time)")
print(f"GPU multiply           {t_gpu:9.1f} ms")
print(f"largest relative difference CPU vs GPU: {diff:.1e}  -> "
      f"{'results match' if diff < 1e-4 else 'MISMATCH'}")
print(f"speedup, multiply only: {t_cpu / t_gpu:6.1f}x")

# TRY IT:
#  1. Delete the torch.cuda.synchronize() right after the multiply (the line marked
#     "wait, so the timing is real"). What happens to the
#     "GPU multiply" time? Is it real?
#  2. Compare "GPU multiply" with the time printed by ../src/05_matmul (the kernel
#     we wrote by hand). Which is faster, and by how much?
