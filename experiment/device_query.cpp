#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No AMD GPU devices found!\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);

        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("Architecture GFX Target:   %s\n", prop.gcnArchName);
        printf("Compute Units (CUs):       %d\n", prop.multiProcessorCount);
        printf("Total Global Memory:       %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        
        printf("\n--- Thread & Block Limits ---\n");
        printf("Max Threads per Block:     %d\n", prop.maxThreadsPerBlock);
        printf("Max Block Dimensions:      [%d, %d, %d]\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions:       [%d, %d, %d]\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
               
        printf("\n--- Warp / Wavefront Size ---\n");
        printf("Wavefront Size (Warp):     %d threads\n", prop.warpSize);
    }

    return 0;
}

