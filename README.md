# GPU Programming hands-on (ICDS Fall Workshop, Nov 13, 2026)

Short CUDA programs to build, run, break, debug and time. Each file starts with its own
instructions and ends with a "TRY IT" list of small experiments. Fixed versions of the exercise
and of the four bugs are in `solutions/` (build them with `make solutions` in the top folder).

## Get started

**On Roar: follow `FOLLOW_ALONG.md`.** It is a list of commands to copy and paste, from logging in
to looking at the pictures. In short:

```bash
salloc --account=open --partition=standard --nodes=1 --ntasks=4 --mem=16G --gres=gpu:p100:1 --time=01:30:00
source env.sh          # CUDA compiler and tools, plus a Python with PyTorch and CuPy
make                   # build everything
bash check_all.sh      # runs every demo once and prints PASS or FAIL
cd src
```

On another machine you need an NVIDIA GPU, the CUDA toolkit (`nvcc`), and, for the Python files,
`pip install torch cupy-cuda12x pillow` (`cupy-cuda12x` is the CuPy build for CUDA 12; pick the one
that matches your CUDA version).

To rebuild after editing a file: `make -C ..` (from inside `src`).

## Files

| Files | What they are |
|---|---|
| `src/*.cu` | The CUDA C++ programs, one per step |
| `python/02_vector_add_cupy.py`, `python/04_image_filters_cupy.py` | The same kernels as the `.cu` files of the same name, launched from Python with CuPy |
| `python/05_matmul_torch.py` | Matrix multiplication with PyTorch: no kernel at all |
| `solutions/` | Fixed versions of the exercise and of the four bugs |
| `photos/earth.jpg` | A photo to try the filters on (NASA, Apollo 17) |

## Part 1. Hello

```bash
./01_hello
```

Change `<<<2, 4>>>`, rebuild, run. Then comment out `cudaDeviceSynchronize()` and try again.

## Part 2. Vector add

```bash
# open 02_vector_add_exercise.cu and fill in the four numbered blanks (the FIXME words), then:
make -C .. src/02_vector_add_exercise && ./02_vector_add_exercise

./02_vector_add             # CPU against GPU, with a timing report
./02_vector_add 1000        # a tiny problem

python3 ../python/02_vector_add_cupy.py    # the same kernel, launched from Python
```

## Part 3. Bug hunt

| Program | Symptom | Tool |
|---|---|---|
| `./03_bug1_silent` | No error, answer is all zeros | error checks (`CHECK`) |
| `./03_bug2_bounds` | No error, answer looks nearly right | `compute-sanitizer ./03_bug2_bounds` |
| `./03_bug3_race` | A different answer every run | think about shared data; `atomicAdd` |
| `./03_bug4_inf` | Result is `inf` | `cuda-gdb -batch -x 03_bug4_inf.gdb ./03_bug4_inf` |

## Part 4. Photo filters, and where the time goes

`photos/earth.jpg` comes with the lesson. You can use your own photo instead (jpg or png), or leave
the name out to use the built-in test picture.
The programs write PNG files named `out_*.png`; open them in VS Code or the portal's file browser.

```bash
./04_image_filters ../photos/earth.jpg                   # grayscale and edge detection
python3 ../python/04_image_filters_cupy.py ../photos/earth.jpg   # the same kernels, from Python

OMP_NUM_THREADS=1 ./04_blur_cpu ../photos/earth.jpg      # blur on the CPU, one core
OMP_NUM_THREADS=4 ./04_blur_cpu ../photos/earth.jpg      # blur on the CPU, four cores
./04_blur_v1 ../photos/earth.jpg                         # first GPU port
./04_blur_v2 ../photos/earth.jpg                         # GPU port with the picture kept on the GPU

# nsys is the command of NVIDIA Nsight Systems, the profiler
nsys profile --stats=true --force-overwrite true -o blur_v1 ./04_blur_v1 ../photos/earth.jpg 100
nsys stats --force-export=true --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum blur_v1.nsys-rep
bash profile_table.sh blur_v1.nsys-rep     # the same two tables, three columns, times in milliseconds

./04_blur_v2 ../photos/earth.jpg 300 1                   # last value = threads per block: try 1, 32, 256, 1024
```

| Program | Total time | Per pass |
|---|---|---|
| CPU, one core | | |
| CPU, four cores | | |
| GPU version 1 | | |
| GPU version 2 | | |

## Part 5. You rarely write the kernel yourself

```bash
./05_matmul                                # matrix multiply with a hand-written kernel
python3 ../python/05_matmul_torch.py       # the same with PyTorch (needs PyTorch with GPU support)
```

On Roar, `source env.sh` provides a Python with PyTorch and CuPy.

## Checklists

Before you trust a GPU result: it matches the CPU answer; every CUDA call is checked;
`compute-sanitizer` reports 0 errors; repeated runs agree.

Before you quote a speedup: the timer waited for the GPU; copies are counted; the CPU baseline is
an optimised build on more than one core; the GPU was not shared.

## Where to go next

Oak Ridge / NVIDIA CUDA Training Series: https://www.olcf.ornl.gov/cuda-training-series/
(exercises: https://github.com/olcf/cuda-training-series).
