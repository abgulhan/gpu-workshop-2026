# Scripted cuda-gdb session for 03_bug4_inf.
# Run with:   cuda-gdb -batch -x 03_bug4_inf.gdb ./03_bug4_inf
set pagination off
set confirm off

echo \n===== 1. Stop inside the kernel, on the line that divides =====\n
break 03_bug4_inf.cu:51
run

echo \n===== 2. What is running on the GPU, and where is every thread? =====\n
info cuda kernels
info cuda threads

echo \n===== 3. The debugger is looking at ONE thread. Which element does it own? =====\n
print i
print before[i]
print after[i]

echo \n===== 4. Jump to another thread (block 2, thread 5) and look again =====\n
cuda block 2 thread 5
print i
print before[i]
print after[i]

echo \n===== 5. Carry on, and stop on the next line only in a thread whose result r is huge =====\n
break 03_bug4_inf.cu:52 if r > 1000000
continue

echo \n===== 6. The culprit =====\n
cuda kernel block thread
print i
print r
print before[i]
print after[i]
kill
quit
