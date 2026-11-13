#!/bin/bash
# check_all.sh -- run every demo once and say PASS or FAIL for each. One to two minutes on one GPU.
# Use it before presenting, on the machine and the GPU you will present from:
#     make && bash check_all.sh
# The four "bug" programs are SUPPOSED to misbehave; PASS there means "the bug shows as planned".
cd "$(dirname "$0")/src" || exit 1
PHOTO=../photos/earth.jpg
pass=0; fail=0
check() {   # check "name" "text that must appear" command...
    local name=$1 want=$2; shift 2
    local out; out=$("$@" 2>&1)
    if echo "$out" | grep -q -- "$want"; then echo "PASS  $name"; pass=$((pass+1))
    else echo "FAIL  $name   (wanted: $want)"; echo "$out" | tail -5 | sed 's/^/        /'; fail=$((fail+1)); fi
}
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ */GPUs visible: /'
check "1  hello"                          "block 1, thread 3"       ./01_hello
check "2  vector add"                     "results match"           ./02_vector_add
check "3  bug 1 shows (all zeros)"        "out\[3\]    = 0 "        ./03_bug1_silent
check "3  bug 2 found by memory checker"  "ERROR SUMMARY: [1-9]"    compute-sanitizer ./03_bug2_bounds
check "3  bug 3 shows (wrong count)"      "WRONG"                   ./03_bug3_race
check "3  bug 4 shows (inf)"              "mean ratio = inf"        ./03_bug4_inf
check "3  bug 4 found by debugger"        "= 1234"                   cuda-gdb -batch -x 03_bug4_inf.gdb ./03_bug4_inf
check "4  photo filters"                  "wrote out_edges.png"     ./04_image_filters $PHOTO
check "4  blur, CPU"                      "check value"             env OMP_NUM_THREADS=4 ./04_blur_cpu $PHOTO 100
check "4  blur, GPU version 1"            "check value"             ./04_blur_v1 $PHOTO 100
check "4  blur, GPU version 2"            "check value"             ./04_blur_v2 $PHOTO 100
check "4  profiler (nsys)"                "blur_pass"                nsys profile --stats=true -o /tmp/check_all_$$ ./04_blur_v1 $PHOTO 20
rm -f /tmp/check_all_$$.*
check "5  hand-written matrix multiply"   "results match"           ./05_matmul 2048
if python3 -c "import torch" 2>/dev/null; then
    check "5  PyTorch matrix multiply"    "results match"           python3 ../python/05_matmul_torch.py 2048
else
    echo "SKIP  5  PyTorch matrix multiply   (this python3 has no PyTorch)"
fi
if python3 -c "import cupy" 2>/dev/null; then
    check "2  vector add from Python (CuPy)"     "results match"          python3 ../python/02_vector_add_cupy.py 4000000
    check "4  photo filters from Python (CuPy)"  "wrote out_edges_py.png" python3 ../python/04_image_filters_cupy.py $PHOTO
else
    echo "SKIP  2 and 4 from Python (CuPy)   (this python3 has no CuPy)"
fi
v=$(for p in "env OMP_NUM_THREADS=4 ./04_blur_cpu" ./04_blur_v1 ./04_blur_v2; do $p $PHOTO 100 2>&1 | grep "check value"; done | sort -u | wc -l)
if [ "$v" = 1 ]; then echo "PASS  4  the three blur programs agree"; pass=$((pass+1)); else echo "FAIL  4  the three blur programs disagree"; fail=$((fail+1)); fi
echo; echo "$pass passed, $fail failed"
