#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"

# define N 100

// =====================================================================
/*
This CUDA code computes the squares from 0 to N-1 and stores them in an array ('arr'). 
*/
// =====================================================================

// CUDA kernel function: save_num

__global__ void save_num(int * arr) {
    arr[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {

    int arr[N];   // Declare an array in cpu.
    int *gpu_pointer;  // This pointer will be used to store a gpu memory address.

    /*
    cudaMalloc is a function in CUDA programming that allocates memory on the GPU. It is used to dynamically allocate memory in the GPU's global memory space, which can be accessed and utilized by CUDA kernels running on the GPU. It is analogous to malloc but for GPUs.
    */
    cudaMalloc(&gpu_pointer, N * sizeof(int));

    // Launch the cuda kernal function with 1 block and N threads per block.
    save_num<<<1, N>>>(gpu_pointer);
    
    /*
    cudaMemcpy is a function in CUDA programming that enables data transfer between the CPU (host) and the GPU (device). It allows you to copy data from one memory location to another, such as from the CPU's memory to the GPU's memory or vice versa.
    */
    cudaMemcpy(arr, gpu_pointer, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the values stored in arr.
    for (int i =0; i<N; i++) {
        printf("%d: arr[%d]\n", i, arr[i]);
    }

    /*
    Note: The cudaMemcpy function is blocking, which means that it will not return until the data transfer is finished. When cudaMemcpy returns, you can safely assume that the data has been copied to the destination memory location (either from host to device or vice versa). Hence, we need not use cudaDeviceSynchronize.
    */

    return 0;
}