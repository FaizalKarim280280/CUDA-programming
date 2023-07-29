#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"

// =====================================================================
/*
 Basic CUDA program that prints "Hello world" to the console
*/
// =====================================================================

// CUDA kernel function: hello
// This function will be executed on the GPU
__global__ void hello()
{
    // The printf function is not recommended inside a CUDA kernel.
    // It is used here for demonstration purposes only.
    // In a real CUDA application, printing from a kernel can cause issues and may not work as expected.
    // It can lead to unpredictable output and performance degradation due to serialization.
    printf("Hello world\n");
}

// Main function: Entry point of the program
int main()
{
    // Launch the CUDA kernel function 'hello' with 1 block and 1 thread per block
    // <<<1, 1>>> is called the execution configuration. It specifies how many blocks and threads to use.
    hello<<<1, 1>>>();

    // cudaDeviceSynchronize is used to ensure that the CUDA kernel execution is completed before continuing.
    // This function blocks the CPU until the GPU completes all its tasks.
    cudaDeviceSynchronize();

    // Note: In real-world CUDA applications, you may need to check for CUDA errors using cudaGetLastError() or other error handling mechanisms.
    return 0;
}
