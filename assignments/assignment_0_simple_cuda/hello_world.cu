#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_world_fun() {
    printf("Hello world from GPU\n");
}

int main() {
    // Launching kernel to run this function on the GPU
    hello_world_fun<<<1, 1>>>();
    
    // Check for errors after the kernel launch
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }

    // This call blocks the CPU main thread until the GPU operations complete.
    cudaDeviceSynchronize();
    
    // // Check for errors after synchronization
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error after sync: " << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }
    
    std::cout << "Hello world from CPU" << std::endl;
    
    return 0;
}
