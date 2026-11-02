import numpy as np
import torch
from timeit import default_timer as timer

# 1. GPU Function (Replaces the custom @cuda.jit kernel)
# PyTorch automatically vectorizes this operation into an optimized ROCm kernel under the hood.
def increment_by_one_gpu(an_array_tensor):
    an_array_tensor += 1

# 2. CPU Function
def increment_by_one_nogpu(an_array):
    pos = 0
    while pos < an_array.size:
        an_array[pos] += 1
        pos += 1

if __name__ == "__main__":
    # Detect AMD GPU via ROCm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})\n")

    # Create NumPy array of 10 million 1s
    n = 10_000_000
    arr = np.ones(n, dtype=np.float32)

    # Move the array to AMD GPU memory (Equivalent to cuda.to_device)
    d_arr = torch.from_numpy(arr.copy()).to(device)
    if device == "cuda":
        torch.cuda.synchronize()

    # Print initial array values
    print("Initial GPU Array Data:     ", arr[:5], "... (truncated)")
    print("Initial NON-GPU Array Data: ", arr[:5], "... (truncated)\n")

    # --- GPU EXECUTION ---
    # Warm-up run to initialize the GPU kernel before timing
    _temp = d_arr + 1
    if device == "cuda":
        torch.cuda.synchronize()

    start = timer()
    increment_by_one_gpu(d_arr)
    # Block Python until the AMD GPU completely finishes executing the operation
    if device == "cuda":
        torch.cuda.synchronize()
    dt_gpu = timer() - start

    print(f"Time With AMD GPU: {dt_gpu:.6f} seconds")
    
    # --- CPU EXECUTION ---
    start = timer()
    increment_by_one_nogpu(arr)
    dt_cpu = timer() - start

    print(f"Time Without GPU:  {dt_cpu:.6f} seconds")

    # --- ADVANTAGE ---
    print(f"\n🚀 GPU Speedup Factor: {dt_cpu / dt_gpu:.2f}x faster\n")

    # Move data back to host memory (Equivalent to copy_to_host)
    gpu_arr = d_arr.cpu().numpy()

    # Print final verified array values
    print("Final GPU Array Result:     ", gpu_arr[:5], "... (all entries should be 2.0)")
    print("Final NON-GPU Array Result: ", arr[:5], "... (all entries should be 2.0)")
