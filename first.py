import math
import time
import numpy as np
import torch

# 1. Initialize data on the CPU
print("Generating 30 million random numbers...")
random_array = np.random.random(30_000_000).astype(np.float32)

# Verify AMD GPU availability via ROCm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# 2. Transfer data to the AMD GPU (Equivalent to cuda.to_device)
print("\nMoving data to AMD GPU...")
start_transfer = time.perf_counter()
gpu_random_array = torch.from_numpy(random_array).to(device)
if device == "cuda":
    torch.cuda.synchronize()
print(f"Transfer time: {time.perf_counter() - start_transfer:.4f} seconds")

# 3. Define the operation (Equivalent to your @cuda.jit kernel)
# PyTorch compiles elementwise operations natively into GPU grid kernels under the hood
def gpu_foo(input_tensor):
    return input_tensor * 2

# Warm-up run to initialize the GPU kernel before timing
_ = gpu_foo(gpu_random_array)
if device == "cuda":
    torch.cuda.synchronize()

# 4. Measure execution time (Replacing %%timeit -n 100)
print("\nTiming execution over 100 loops...")
runs = 100
durations = []

for _ in range(runs):
    start_loop = time.perf_counter()
    
    # Run the kernel operation
    gpu_output = gpu_foo(gpu_random_array)
    
    # Synchronize ensures the AMD GPU finishes computing before the clock stops
    if device == "cuda":
        torch.cuda.synchronize()
        
    durations.append(time.perf_counter() - start_loop)

avg_duration_ms = (sum(durations) / runs) * 1000
print(f"Average execution time: {avg_duration_ms:.4f} ms per loop")

# 5. Bring data back to CPU host (Equivalent to copy_to_host)
print("\nCopying data back to host...")
output = gpu_output.cpu().numpy()

print(f"Input Sample:  {random_array[:5]}")
print(f"Output Sample: {output[:5]}")
