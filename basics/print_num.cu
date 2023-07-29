#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"

#define N 100

// ===================================================================================
/*
This CUDA code prints thread indices (0 to 99) using 100 parallel threads on the GPU. 
*/
// ===================================================================================


// CUDA kernel function: print_num
// This function prints the thread index (threadIdx.x) to the console.
__global__ void print_num()
{
    printf("%d\n", threadIdx.x);
}

// Main function: Entry point of the program
int main()
{
    // Launch the CUDA kernel function 'print_num' with 1 block and N threads per block
    // <<<1, N>>> is called the execution configuration. It specifies how many blocks and threads to use.
    // In this case, the kernel will be executed by 1 block, and each block will have N (100) threads.
    // The total number of threads executing the kernel will be 1 * N = 100.
    print_num<<<1, N>>>();

    cudaDeviceSynchronize();

    return 0;
}
