#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(error), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU Kernel: Each thread calculates one specific element (row, col) of Matrix C
__global__ void matrixMul(const float* A, const float* B, float* C, int N) {
    // Calculate global Row and Column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety boundary check
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 4; // Let's use a 4x4 matrix for easy verification
    size_t size = N * N * sizeof(float);

    // 1. Allocate CPU Memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize sample data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; // Matrix full of 1s
        h_B[i] = 2.0f; // Matrix full of 2s
    }

    // 2. Allocate GPU Memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size));
    CHECK_HIP(hipMalloc(&d_B, size));
    CHECK_HIP(hipMalloc(&d_C, size));

    // 3. Copy data to GPU
    CHECK_HIP(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    // 4. Define Thread Block and Grid structure
    // We create 2x2 blocks of threads. Total threads per block = 4.
    dim3 threadsPerBlock(2, 2); 
    
    // Calculate blocks needed to cover the NxN matrix
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 5. Launch the 2D Kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 6. Sync and pull data back
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    // 7. Print results
    printf("Result Matrix C (should be all 8.0 for a 4x4 of 1s * 2s):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}

