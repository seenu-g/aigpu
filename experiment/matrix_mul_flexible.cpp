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

// GPU Kernel: Works with non-square matrices
// A is (A_rows x A_cols), B is (A_cols x B_cols), C is (A_rows x B_cols)
__global__ void matrixMul(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    // Calculate which row and column of the output matrix C this thread handles
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety boundary check: Make sure thread stays within the boundaries of Matrix C
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        // The inner dimension loop runs up to A_cols (which must equal B_rows)
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

int main() {
    // Define sizes for a 4x4 multiplied by a 4x40 matrix
    int A_rows = 4;
    int A_cols = 4; // This must equal B_rows
    int B_rows = 4;
    int B_cols = 40;

    // Output Matrix C will have dimensions: A_rows x B_cols (4 x 40)
    int C_rows = A_rows;
    int C_cols = B_cols;

    size_t size_A = A_rows * A_cols * sizeof(float);
    size_t size_B = B_rows * B_cols * sizeof(float);
    size_t size_C = C_rows * C_cols * sizeof(float);

    // 1. Allocate CPU Memory
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    // Initialize sample data
    for (int i = 0; i < A_rows * A_cols; i++) h_A[i] = 1.0f; // Matrix A full of 1s
    for (int i = 0; i < B_rows * B_cols; i++) h_B[i] = 2.0f; // Matrix B full of 2s

    // 2. Allocate GPU Memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size_A));
    CHECK_HIP(hipMalloc(&d_B, size_B));
    CHECK_HIP(hipMalloc(&d_C, size_C));

    // 3. Copy data to GPU
    CHECK_HIP(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));

    // 4. Define Thread Block and Grid structure
    // A 16x16 block layout is highly efficient for your Radeon's 32-thread wavefronts
    dim3 threadsPerBlock(16, 16); 
    
    // Calculate enough blocks to cover the dimensions of the final Matrix C (4 x 40)
    dim3 numBlocks((C_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (C_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 5. Launch the Kernel passing all unique dimensions
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);

    // 6. Sync and pull data back
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost));

    // 7. Print results (just showing the first few columns to verify)
    printf("Input Matrix A dimensions: %d x %d\n", A_rows, A_cols);
    printf("Input Matrix B dimensions: %d x %d\n", B_rows, B_cols);
    printf("Result Matrix C dimensions: %d x %d\n", C_rows, C_cols);
    printf("Printing first 5 columns of the result:\n");

    for (int i = 0; i < C_rows; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%5.1f ", h_C[i * C_cols + j]);
        }
        printf("... [plus %d more columns]\n", C_cols - 5);
    }

    // Clean up
    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}

