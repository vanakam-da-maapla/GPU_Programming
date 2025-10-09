#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_fun(int *arr) {
    int threadIndex = threadIdx.x;
    arr[threadIndex] = threadIndex * threadIndex;
}

int main() {

    int N = 256;
    std::cout << "Enter Array count: ";
    std::cin >> N;
    int *arr = (int*) malloc(N * sizeof(int));
    int *d_arr;

    //
    cudaMalloc(&d_arr, N * sizeof(int));

    //
    square_fun<<<1, N>>>(d_arr);

    //
    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++){
      std::cout << arr[i] << "\n";
    }

    // Free the space allocated in GPU RAM
    cudaFree(d_arr);

    // Free the space allocated in CPU RAM
    free(arr);
    
    return 0;
}