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

// GPU Kernel: Every thread handles exactly one pixel destination
__global__ void grayscalePPM(const unsigned char* input, unsigned char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // PPM data has 3 color channels (RGB) per pixel
        int idx = (row * width + col) * 3;

        unsigned char r = input[idx + 0];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        // Apply grayscale conversion weights
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        output[idx + 0] = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;
    }
}

int main() {
    FILE* infile = fopen("input.ppm", "r");
    if (!infile) {
        printf("Error: Cannot open input.ppm!\n");
        return 1;
    }

    // Parse the simple PPM header info
    char format[3];
    int width, height, max_val;
    if (fscanf(infile, "%2s %d %d %d", format, &width, &height, &max_val) != 4) {
        printf("Error reading PPM header\n");
        fclose(infile);
        return 1;
    }

    size_t size = width * height * 3 * sizeof(unsigned char);
    unsigned char* h_input = (unsigned char*)malloc(size);
    unsigned char* h_output = (unsigned char*)malloc(size);

    // Read pixel values from file into CPU memory
    for (int i = 0; i < width * height * 3; i++) {
        int val;
        if (fscanf(infile, "%d", &val) != 1) break;
        h_input[i] = (unsigned char)val;
    }
    fclose(infile);

    // Allocate VRAM on AMD Radeon GPU
    unsigned char *d_input, *d_output;
    CHECK_HIP(hipMalloc(&d_input, size));
    CHECK_HIP(hipMalloc(&d_output, size));

    // Copy data: CPU -> GPU
    CHECK_HIP(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));

    // Configure layout blocks
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Run the lock-free pixel processing grid
    grayscalePPM<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Sync and retrieve data: GPU -> CPU
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost));

    // Save out the new PPM image data file
    FILE* outfile = fopen("output_grayscale.ppm", "w");
    fprintf(outfile, "P3\n%d %d\n%d\n", width, height, max_val);
    for (int i = 0; i < width * height; i++) {
        fprintf(outfile, "%d %d %d  ", h_output[i*3], h_output[i*3+1], h_output[i*3+2]);
        if ((i + 1) % width == 0) fprintf(outfile, "\n");
    }
    fclose(outfile);
    printf("Success! Processed offline matrix and saved to output_grayscale.ppm\n");

    // Clean up
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}

