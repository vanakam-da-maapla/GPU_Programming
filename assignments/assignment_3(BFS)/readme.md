# Assignment 3: Parallel BFS Optimization with CUDA

This project implements and optimizes a parallel Breadth-First Search (BFS) on a k-NN graph using CUDA.

This README provides the basic steps to build and run the code.

## 1. How to Build

First, you must build the executables.

1. **Modify `build.sh` (if required!)**

   * Open the `build.sh` file.

   * Find the line: `CUDA_ARCH="sm_90"`

   * Replace `"sm_90"` to match your GPU's "Compute Capability".

2. **Run the Build Script**

   * Make the script executable:

     ```
     chmod +x build.sh
     
     ```

   * Run the script:

     ```
     ./build.sh
     
     ```

This will generate three executables:

* `bfs_baseline`: **(Task 1)** The baseline implementation.

* `bfs_stream`: **(Task 2)** The version optimized with CUDA Streams.

* `bfs_graph`: **(Task 3)** The version optimized with CUDA Graphs.

## 2. How to Run

To run all the three tasks at once, there is script file called `bfs.sh` which runs all the task executables

1. **Executing `bfs.sh`:**

    ```
    ./bfs.sh <absolute path to the graph> <absolute path to dataset file i.e. sift_base.fvecs> <node#>
        
    ```

