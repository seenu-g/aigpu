#include <hip/hip_runtime.h>  
#include <stdio.h>

// In GPU computing, a Block (or Thread Block) is a programming group of threads that execute the same kernel code.
// A Thread is an individual worker.
// A Block is a specific workspace or assembly line table.
// The Grid is the entire factory floor (all the blocks combined).

__global__ void hello() {
   /* printf("Hello World from Thread %d!\n", threadIdx.x); */
       printf("Hello from Block %d, Thread %d\n", blockIdx.x, threadIdx.x);

}

int main() {
    // Launching the kernel on the AMD GPU
    // hello<<< 1, 4 >>>();
    hello<<< 3, 4 >>>();

    // Catch the return value to satisfy the compiler
    hipError_t err = hipDeviceSynchronize(); 
    
    if (err != hipSuccess) {
        printf("GPU Error: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    return 0;
}

