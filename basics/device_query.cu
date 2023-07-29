#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"
#include "cuda_runtime.h"

/*
Device query refers to the process of gathering information about the available GPU devices on a computer system. In the context of CUDA programming, device query involves using CUDA API functions to retrieve and display various properties and capabilities of the GPUs present in the system.
*/


// Function to retrieve and display information about available GPU devices
int getDevice()
{
    int device = 0;
    // Get the number of available CUDA-capable devices on the system
    cudaGetDeviceCount(&device);

    // If there are no GPUs available, print a message and return 0
    if (device == 0)
    {
        printf("No GPU available on this system.\n");
        return 0;
    }

    // Loop through all available GPU devices
    for (int i = 0; i < device; i++)
    {
        cudaDeviceProp prop;
        // Get the properties of the i-th GPU
        cudaGetDeviceProperties(&prop, i);

        // Display information about the GPU
        printf("Device %d:\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Number of MultiProcessors: %d\n", prop.multiProcessorCount);
        printf("Global Memory Size: %.2f MB\n", static_cast<float>(prop.totalGlobalMem) / (1024.0 * 1024.0));
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Clock Rate: %d KHz\n", prop.clockRate);
        printf("Memory Clock Rate: %d KHz\n", prop.memoryClockRate);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("Shared Memory per Block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("Maximum Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum Block Dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Compute Mode: %s\n", prop.computeMode == cudaComputeModeDefault ? "Default (Multiple threads can use CUDA)" : "Exclusive (Only one thread can use CUDA)");
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Supported" : "Not supported");
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Supported" : "Not supported");
        printf("Async Engine Count: %d\n", prop.asyncEngineCount);
        printf("GPU Overlap: %s\n", prop.deviceOverlap ? "Supported" : "Not supported");
        printf("ECC Memory: %s\n", prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("Compute Preemption: %s\n", prop.computePreemptionSupported ? "Supported" : "Not supported");
        printf("Kernel Execution Timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
        printf("Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");
        printf("Can Map Host Memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}

// Main function
int main()
{
    // Call the getDevice() function to display GPU information
    getDevice();

    return 0;
}
