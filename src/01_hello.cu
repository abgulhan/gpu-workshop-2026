// 01_hello.cu -- your first GPU program.
//
// Build:  make -C ..        (or: nvcc -o 01_hello 01_hello.cu)
// Run:    ./01_hello
#include <cstdio>

// "__global__" marks a KERNEL: a function that runs on the GPU.
// Every GPU thread runs its own copy of this function at the same time.
// blockIdx.x = the number of my block, threadIdx.x = my number inside the block.
// Both are built in. (".x" because blocks can also be laid out in 2-D or 3-D; we only use x.)
__global__ void hello()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch the kernel on the GPU: 2 blocks, 4 threads in each block
    // = 8 copies of hello() running in parallel.
    hello<<<2, 4>>>();

    // The launch returns immediately (the GPU works in the background).
    // Wait here until the GPU is done, otherwise the program may end
    // before anything is printed.
    cudaDeviceSynchronize();

    printf("Back on the CPU.\n");
    return 0;
}

// TRY IT:
//  1. Change <<<2, 4>>> to <<<3, 2>>>. How many lines do you get?
//  2. Try <<<1, 8>>> and <<<8, 1>>>. Both give 8 lines. What is different?
//  3. Comment out cudaDeviceSynchronize(). What happens, and why?
//
// After every change:  make -C ..   (or the nvcc line at the top), then run again.
// Note: the lines usually appear in a tidy order, because the GPU collects the text
// and hands it over when the kernel has finished. The threads themselves run at the
// same time, so never write a program that depends on their order.
