#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro to check HIP operations and report errors instantly
#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(error), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU Kernel: Every thread is assigned to exactly ONE output pixel matrix element
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    // Result-First Tracking: Calculate unique global 2D coordinates for this worker
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary Check: Ensure threads staying outside the 2048x1355 canvas don't execute
    if (col < width && row < height) {
        // Find the precise memory location for this pixel (3 color channels: RGB)
        int idx = (row * width + col) * 3;

        unsigned char r = input[idx + 0];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        // Apply standard luminance coefficients for human visual perception
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        // Write the calculated result safely into its isolated VRAM bucket
        // output[idx + 0] = gray;
        // output[idx + 1] = gray;
        // output[idx + 2] = gray;
	
	// Replace your grayscale lines with these inversion lines:
        output[idx + 0] = 255 - r; // Invert Red
        output[idx + 1] = 255 - g; // Invert Green
        output[idx + 2] = 255 - b; // Invert Blue
				   
	// Let's sample a small 4x4 matrix of pixels right in the middle of the image
        // For a 2048x1355 image, center is roughly around row 670, column 1024
        if (col >= 1024 && col < 1028 && row >= 670 && row < 674) {
            printf("Destination Pixel Matrix Box [%d, %d] -> Processed by Global Block [%d, %d], Thread [%d, %d]\n", 
                   col, row, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        }

    }
}

// Built-in BMP File Writer: Encodes raw CPU pixel buffers into a viewable image file
void saveBMP(const char* filename, const unsigned char* rgb, int w, int h) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    int filesize = 54 + 3 * w * h;
    
    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);
    
    bmpinfoheader[4] = (unsigned char)(w);
    bmpinfoheader[5] = (unsigned char)(w >> 8);
    bmpinfoheader[6] = (unsigned char)(w >> 16);
    bmpinfoheader[7] = (unsigned char)(w >> 24);
    
    bmpinfoheader[8] = (unsigned char)(h);
    bmpinfoheader[9] = (unsigned char)(h >> 8);
    bmpinfoheader[10] = (unsigned char)(h >> 16);
    bmpinfoheader[11] = (unsigned char)(h >> 24);
    
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    
    // BMP files store image rows from bottom-to-top layout
    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            int p = (i * w + j) * 3;
            // Write colors in expected Blue, Green, Red order
            fwrite(&rgb[p+2], 1, 1, f);
            fwrite(&rgb[p+1], 1, 1, f);
            fwrite(&rgb[p+0], 1, 1, f);
        }
    }
    fclose(f);
}

int main() {
    // Configured exactly to your target dimensions from FFmpeg
    int width = 2048;  
    int height = 1355; 
    
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    
    // Allocate Host (CPU) memory buffers
    unsigned char* h_input = (unsigned char*)malloc(imageSize);
    unsigned char* h_output = (unsigned char*)malloc(imageSize);

    // Read the binary stream parsed out by FFmpeg
    FILE* f = fopen("raw_input.bin", "rb");
    if (!f) {
        printf("Error: Could not open raw_input.bin! Check if the FFmpeg command finished.\n");
        free(h_input); free(h_output);
        return 1;
    }
    size_t read_bytes = fread(h_input, 1, imageSize, f);
    fclose(f);

    // Allocate Device (GPU) VRAM space on your Radeon card
    unsigned char *d_input, *d_output;
    CHECK_HIP(hipMalloc(&d_input, imageSize));
    CHECK_HIP(hipMalloc(&d_output, imageSize));

    // Copy raw data matrix across the PCIe pipeline: CPU -> GPU
    CHECK_HIP(hipMemcpy(d_input, h_input, imageSize, hipMemcpyHostToDevice));

    // Define 16x16 geometry blocks (256 threads, packs perfectly into 32-thread wavefronts)
    dim3 threadsPerBlock(16, 16);
    
    // Calculate precise block dimensions required to span the entire output image matrix
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching parallel pixel processor grid: %d x %d blocks...\n", numBlocks.x, numBlocks.y);

    // Execute the parallel lock-free image processing matrix grid
    grayscaleKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Block CPU thread execution until the GPU has processed all millions of cells
    CHECK_HIP(hipDeviceSynchronize());
    
    // Retrieve processed results back across the pipeline: GPU -> CPU
    CHECK_HIP(hipMemcpy(h_output, d_output, imageSize, hipMemcpyDeviceToHost));

    // Serialize output buffer as a standard uncompressed image file
    saveBMP("output_result.bmp", h_output, width, height);
    printf("Success! Processed 2.7+ million pixels and wrote out to output_result.bmp\n");

    // Clear and free all VRAM slots to avoid leaks
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}

