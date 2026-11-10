#include <hip/hip_runtime.h>
#include <stdio.h>

// The GPU Kernel: now logs which thread index did the work
__global__ void vectorAdd(const float* A, const float* B, float* C, int* thread_log, int N) {
    // Calculate the global index for this thread
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (i < N) {
        C[i] = A[i] + B[i];
        thread_log[i] = i; // Record the global thread index that handled this element
    }
}

int main() {
    int N = 12; 
    size_t size_float = N * sizeof(float);
    size_t size_int = N * sizeof(int);

    // 1. Allocate memory on the CPU
    float h_A[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float h_B[12] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    float h_C[12] = {0}; 
    int h_thread_log[12] = {0}; // To store the thread ID mapping

    // 2. Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    int *d_thread_log;
    hipMalloc(&d_A, size_float);
    hipMalloc(&d_B, size_float);
    hipMalloc(&d_C, size_float);
    hipMalloc(&d_thread_log, size_int);

    // 3. Copy input data to GPU
    hipMemcpy(d_A, h_A, size_float, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_float, hipMemcpyHostToDevice);

    // 4. Launch the kernel: 3 blocks, 4 threads per block
    vectorAdd<<<3, 4>>>(d_A, d_B, d_C, d_thread_log, N);

    // 5. Synchronize and copy results back
    hipDeviceSynchronize();
    hipMemcpy(h_C, d_C, size_float, hipMemcpyDeviceToHost);
    hipMemcpy(h_thread_log, d_thread_log, size_int, hipMemcpyDeviceToHost);

    // 6. Print out the results showing the Thread Index mapping
    printf("%-15s | %-20s | %-15s\n", "Array Index", "Math Operation", "Worked By Global Thread");
    printf("-------------------------------------------------------------------------\n");
    for (int i = 0; i < N; i++) {
        printf("Element [%2d]     | %4.1f + %4.1f = %4.1f   | Global Thread Index: %d\n", 
               i, h_A[i], h_B[i], h_C[i], h_thread_log[i]);
    }

    // 7. Clean up
     
    (void)hipFree(d_A); 
    (void)hipFree(d_B); 
    (void)hipFree(d_C); 
    (void)hipFree(d_thread_log);
    return 0;
}

