# Follow along on Roar: copy and paste

Every grey box below can be pasted into a terminal as it is. Lines starting with `#` are comments.
You need a Penn State account with access to Roar. Nothing is installed on your own computer.

## Step 1. Log in

```bash
ssh YOUR_PSU_ID@submit.hpc.psu.edu
```

Replace `YOUR_PSU_ID` (for example `abc1234`).

- The first time, ssh asks `Are you sure you want to continue connecting?` Type `yes`.
- Type your password. Nothing appears on the screen while you type; that is normal.
- Approve the two-factor request on your phone.
- No terminal on your computer? Use https://portal.hpc.psu.edu and open "Clusters", then
  "Shell Access".

You are now on a **login node**: a shared entry machine with no GPU. Do not run the demos here.

## Step 2. Get the files (5 seconds)

```bash
cp -r /scratch/abg6029/gpu_workshop ~/gpu_workshop
```

Do this once. If `~/gpu_workshop` already exists, skip this step (or remove the old copy first
with `rm -rf ~/gpu_workshop`).

## Step 3. Ask for a GPU (usually under a minute)

```bash
salloc --account=open --partition=standard --nodes=1 --ntasks=4 --mem=16G --gres=gpu:p100:1 --time=01:30:00
```

`salloc: Pending job allocation ...` means you are in the queue. Wait for the line
`Nodes ... are ready for job`. You are now on a machine with one GPU, for one and a half hours, paid from
your free credits (about 0.03 of the 3 credits that every user gets each month). Check it:

```bash
nvidia-smi
```

It prints a table with one GPU (a Tesla P100).

If the queue takes more than two minutes, press Ctrl-C and ask for another GPU type: replace `p100`
with `v100`, then `a40`. (The workshop was tested on all three.)

## Step 4. Set up and build (about 1 minute)

```bash
cd ~/gpu_workshop
source env.sh
make
```

`source env.sh` loads the CUDA compiler and a ready-made Python with PyTorch and CuPy. It prints
two lines, one starting with `CUDA:` and one with `Python:`.
`make` builds whatever is not built yet and prints one line per program. If it says
`Nothing to be done for 'all'`, everything is already built; carry on.

**Run `cd ~/gpu_workshop && source env.sh` again whenever you start a new session.**

## Step 5. Check that everything works (1 to 2 minutes)

```bash
bash check_all.sh
```

You should see a list of `PASS` lines ending in `17 passed, 0 failed`. The four "bug" programs are
meant to misbehave; PASS there means the bug shows up as planned. (If the Python environment is
missing you get `14 passed, 0 failed` and two `SKIP` lines; the C++ programs all work.)

## Step 6. Run the demos

```bash
cd ~/gpu_workshop/src

# Part 1. Hello
./01_hello
make -C .. && ./01_hello                          # after you change a file: rebuild, then run again

# Part 2. Vector add
./02_vector_add
python3 ../python/02_vector_add_cupy.py           # the same kernel, launched from Python

# Part 3. Bug hunt
./03_bug1_silent                                  # prints zeros, and no error
./03_bug2_bounds                                  # looks nearly right ...
compute-sanitizer ./03_bug2_bounds                # ... until the memory checker looks at it
./03_bug3_race; ./03_bug3_race; ./03_bug3_race    # three runs, three different answers
./03_bug4_inf                                     # the result is inf
cuda-gdb -batch -x 03_bug4_inf.gdb ./03_bug4_inf  # the debugger finds the thread that did it

# Part 4. Photo filters, and where the time goes (each blur run takes a few seconds)
./04_image_filters ../photos/earth.jpg
python3 ../python/04_image_filters_cupy.py ../photos/earth.jpg     # the same kernels, from Python
OMP_NUM_THREADS=1 ./04_blur_cpu ../photos/earth.jpg               # blur on the CPU, one core
OMP_NUM_THREADS=4 ./04_blur_cpu ../photos/earth.jpg               # four cores
./04_blur_v1 ../photos/earth.jpg
# the profiler (NVIDIA Nsight Systems, command nsys) shows where version 1 spends its time
nsys profile --stats=true --force-overwrite true -o blur_v1 ./04_blur_v1 ../photos/earth.jpg 100
nsys stats --force-export=true --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum blur_v1.nsys-rep
bash profile_table.sh blur_v1.nsys-rep            # the same two tables, three columns, times in milliseconds
./04_blur_v2 ../photos/earth.jpg
./04_blur_v2 ../photos/earth.jpg 300 1            # last value: threads per block (try 1, 32, 256, 1024)

# Part 5. You rarely write the kernel yourself
./05_matmul
python3 ../python/05_matmul_torch.py
```

**The exercise of Part 2.** Open the file, replace each `FIXME`, then build and run it:

```bash
nano 02_vector_add_exercise.cu
make -C .. src/02_vector_add_exercise && ./02_vector_add_exercise
```

In `nano`, save with Ctrl-O then Enter, and leave with Ctrl-X. The same works for every other
file. After you edit any `.cu` file, rebuild with `make -C ..` and run the program again. Python
files need no rebuild. Every file ends with a "TRY IT" list of small experiments.

## Step 7. Look at the pictures

The photo programs write `out_*.png` files into `~/gpu_workshop/src`. Every picture program also
prints a rough preview of its result in the terminal. To see the real pictures:

- **In the browser:** https://portal.hpc.psu.edu, then "Files", then "Home Directory", then
  `gpu_workshop/src`; click a picture.
- **In VS Code on your computer:** install the "Remote - SSH" extension, connect to
  `YOUR_PSU_ID@submit.hpc.psu.edu`, open the folder `gpu_workshop`, click a picture.

## Optional: work from VS Code

1. In VS Code on your computer, install the extension "Remote - SSH" (by Microsoft).
2. Press F1, choose "Remote-SSH: Connect to Host...", and type `YOUR_PSU_ID@submit.hpc.psu.edu`.
   Type your password and confirm the two-factor request.
3. Menu Terminal, New Terminal. Run Steps 2 to 4 of this page in that terminal.
4. Menu File, Open Folder, choose `gpu_workshop`. Click a file to read or edit it; click a picture
   to see it.

ICDS describes this at https://docs.icds.psu.edu/software/visual-studio-code

## Optional: look at the profile on your own computer

The profiler line in Step 6 writes the file `blur_v1.nsys-rep`. Copy it, and the pictures, to your
computer. Run these two lines in a terminal **on your computer**, not on Roar:

```bash
scp YOUR_PSU_ID@submit.hpc.psu.edu:gpu_workshop/src/blur_v1.nsys-rep .
scp "YOUR_PSU_ID@submit.hpc.psu.edu:gpu_workshop/src/out_*.png" .
```

To open the file, install NVIDIA Nsight Systems (free) from
https://developer.nvidia.com/nsight-systems and choose the installer for your system. Take version
2025.1.3 or newer: the program on your computer must not be older than the `nsys` on Roar that
wrote the file. Then start Nsight Systems (on Linux: `nsys-ui`), choose File, Open, and pick
`blur_v1.nsys-rep`. The timeline shows one bar for every kernel and every copy.

## Step 8. When you are done

```bash
exit
```

This gives the GPU back. (Typing `exit` a second time logs you out of Roar.)

## If something goes wrong

| What you see | What to do |
|---|---|
| `cp: cannot stat '/scratch/abg6029/gpu_workshop'` or `Permission denied` in Step 2 | Check the spelling of the path, then tell an instructor. Meanwhile watch the projector: every result is shown there |
| `nvcc: command not found` | You are in a new session. Run `cd ~/gpu_workshop && source env.sh` |
| `nvidia-smi: command not found`, or every line of `check_all.sh` says FAIL | You are still on the login node, which has no GPU. Do Step 3 first |
| `ModuleNotFoundError: No module named 'cupy'` (or `'torch'`) | Run `cd ~/gpu_workshop && source env.sh`. If it says no environment was found, see the next row |
| `env.sh` says "no workshop environment found" | Build your own Python environment (once, about 10 minutes, inside the GPU session): `bash setup_env.sh`, then `source env.sh` again. Every `./program` works without it; only the three `python3` lines need it |
| `salloc: error: ... Invalid account` | Your free `open` account is not active yet. Watch the projector today, and write to icds@psu.edu afterwards |
| The connection dropped, or your laptop went to sleep | The GPU session is gone. Repeat Steps 1, 3 and 4 (not Step 2: your files are still there) |
| After one and a half hours, commands start to fail | The session has ended. Repeat Steps 3 and 4 |
| `nvcc warning : Support for offline compilation for architectures prior to ...` | Harmless. It appears when you type an `nvcc` line by hand; `make` hides it |
| `WARNING: Device-side CUDA Event completion trace is currently enabled` | Harmless. `nsys` prints it at the start of every profile |
| A program prints `out of memory` or hangs | Type `exit`, then repeat Steps 3 and 4 |
