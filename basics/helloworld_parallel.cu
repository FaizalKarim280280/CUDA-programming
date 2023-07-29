# include "stdio.h"
# include "stdlib.h"
# include "cuda.h"

// =====================================================================
/*
This CUDA code prints "Hello world" using 32 parallel threads on the GPU.
*/
// =====================================================================


// CUDA kernel function: helloworld
// This function will be executed on the GPU in parallel
__global__ void helloworld() {
    printf("Hello world\n");
}

// Main function: Entry point of the program
int main() {
    // Launch the CUDA kernel function 'helloworld' with 1 block and 32 threads per block
    // <<<1, 32>>> is called the execution configuration. It specifies how many blocks and threads to use.
    // In this case, the kernel will be executed by 1 block, and each block will have 32 threads.
    // The total number of threads executing the kernel will be 1 * 32 = 32.
    helloworld<<<1, 32>>>();

    cudaDeviceSynchronize();

    return 0;
}
