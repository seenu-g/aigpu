# GPU Experiments

These C++ examples use HIP to demonstrate GPU discovery, kernel launches,
parallel arithmetic, matrix operations, and image processing.

| File | What it performs |
| --- | --- |
| `device_query.cpp` | Lists detected AMD GPU devices and reports each device's architecture, compute units, global memory, thread/block/grid limits, and wavefront size. |
| `helloworld.cpp` | Launches a HIP kernel with three blocks of four threads and prints each block and thread index from the GPU. |
| `vector_add_gpu.cpp` | Adds two 12-element vectors on the GPU and prints each result with the global thread index that computed it. |
| `matrix_mul.cpp` | Multiplies two 4 × 4 matrices on the GPU using a two-dimensional grid of threads, then prints the result. |
| `matrix_mul_flexible.cpp` | Multiplies non-square matrices on the GPU; the example uses a 4 × 4 matrix and a 4 × 40 matrix. |
| `image_ppm.cpp` | Reads RGB pixels from `input.ppm`, converts them to grayscale on the GPU, and writes `output_grayscale.ppm`. |
| `image_grayscale.cpp` | Reads a 2048 × 1355 raw RGB buffer from `raw_input.bin`, inverts each RGB channel on the GPU, and writes `output_result.bmp`. It also prints thread/block information for a small region of the image. |

The examples require a HIP-capable AMD GPU and a compatible ROCm/HIP
development environment. The image examples expect their input files in the
program's working directory.
