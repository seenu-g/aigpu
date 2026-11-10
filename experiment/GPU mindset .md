# GPU Architecture and Parallel Programming Mindset

## 1. The Core Paradigm Shift

CPU and GPU programming often approach a problem from different directions:

- **CPU (often sequential):** Follow the data through a sequence of operations,
  commonly expressed as nested loops.
- **GPU (data-parallel):** Start with the output elements and define how
  independent work items compute them in parallel.

The output-first view is a useful way to design data-parallel kernels, though
it is not a rule that every GPU algorithm must follow.

## 2. The Result-First Mental Model

For each output element, ask:

> I have an output value at row `x`, column `y`. What inputs and operations are
> needed to compute this value?

In a common GPU kernel, a thread or work item is assigned an output index. It
reads the required input values, computes the result, and writes to that output
location. If each work item writes to a distinct location, those writes do not
need synchronization with one another.

For example, a two-dimensional kernel may map its thread coordinates to a
matrix location:

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

The kernel must still check that `row` and `col` are within the matrix bounds.

## 3. Control Parallelism and Data Parallelism

| Approach | Primary focus | Typical coordination |
| --- | --- | --- |
| **Control parallelism (CPU threads)** | Splitting tasks or actions among workers | Shared state may require locks, atomics, or other synchronization |
| **Data parallelism (GPU work items)** | Applying the same operation across many data elements | Independent output locations can often be written without synchronization |

Data-parallel code can avoid write conflicts when each work item owns a unique
output location. This is not automatically guaranteed: overlapping writes,
shared accumulations, or dependencies between work items may require atomics,
barriers, or other synchronization.

## 4. Analogies and Examples

### The Cookie-Cutter Analogy

- **Sequential (CPU):** Cut shapes from dough one at a time.
- **Parallel (GPU):** Use a frame with many cutters so multiple shapes can be
  cut in parallel.
- **Indexing:** Expressions such as `blockIdx.x * blockDim.x + threadIdx.x`
  identify a work item's position in the execution grid; they do not, by
  themselves, perform the computation.

The number of tasks that run at the same instant depends on the GPU's hardware,
resource availability, and kernel configuration. A large grid is generally
scheduled in groups rather than all at once.

### Matrix and Image Processing

A 12-megapixel image has 12,000,000 pixels. If an operation treats each pixel
independently, it can assign a work item to each pixel:

- **CPU:** A straightforward implementation may use a loop to update pixels one
  after another, though CPUs can also use vector instructions and multiple
  threads.
- **GPU:** Many work items process pixels concurrently, with the grid scheduled
  across the GPU's compute units. The actual speedup depends on the operation,
  memory traffic, hardware, and kernel implementation.
