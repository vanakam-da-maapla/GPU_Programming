#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "fileoperations.h"
#include "cudaExceptionHandler.cu"

using namespace std;

/************************************************************************************************
                        COMMAND TO RUN THIS RPOGRAM

compile: nvcc -arch=sm_80 main_4.cu fileoperations.cpp -o mattrans_tiled
run: ./mattrans_tiled 32 "public_test_cases/matrix_a.csv"

************************************************************************************************/

__global__ void matTranspose(int *matrix, int *matrixTranspose, int rows, int cols, int tileWidth){

    //
    int accThreadNum_x = (blockIdx.y * blockDim.y) + threadIdx.y;
    int accThreadNum_y = (blockIdx.x * blockDim.x) + threadIdx.x;

    //
    extern __shared__ int sharedMem[];
    int *s_matrix = sharedMem;

    // printf("\nmatrixMulTiled debug");
    
    if(accThreadNum_x < rows && accThreadNum_y < cols){
        // printf("\nDevice debug 1: %d", tileIndexMatrix);
        s_matrix[(threadIdx.y * tileWidth) + threadIdx.x] = matrix[(accThreadNum_x * cols) + accThreadNum_y];
        // printf("\nDevice debug 2");
    } else s_matrix[(threadIdx.y * tileWidth) + threadIdx.x] = 0;

    __syncthreads();

    if(accThreadNum_x < rows && accThreadNum_y < cols) {
        matrixTranspose[(accThreadNum_y * rows) + accThreadNum_x] = s_matrix[(threadIdx.y * tileWidth) + threadIdx.x];
    }

    __syncthreads();
}

int main(int argc, char* argv[]){

    int tileWidth = stoi(argv[1]);
    string matrixFilename = argv[2];

    
    cout<<"\nFilename: "<<matrixFilename;

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //get matrix A&B from csv file
    int *matrix, rows, cols;
    setMatrixFromFile(false, matrixFilename, matrix, rows, cols);

    //declare dimensions of a  blocks and threads 
    dim3 gridDim3(ceil(static_cast<double>(rows)/tileWidth), ceil(static_cast<double>(cols)/tileWidth));
    dim3 blockDim3(tileWidth, tileWidth);  
    printf("Host debug 2: %f, %f", ceil(static_cast<double>(rows)/tileWidth), ceil(static_cast<double>(cols)/tileWidth));

    //
    unsigned int matrixSize = rows * cols* sizeof(int);
    int *matrixTranspose = (int*) malloc(matrixSize);
    int *d_matrixTranspose, *d_matrix;

    //
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrixSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrixTranspose, matrixSize));

    //
    cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);

    //
    cudaEventRecord(start);
    matTranspose<<<gridDim3, blockDim3, (tileWidth * tileWidth * sizeof(int))>>>(d_matrix, d_matrixTranspose, rows, cols, tileWidth);
    cudaEventRecord(stop);

    //
    cudaMemcpy(matrixTranspose, d_matrixTranspose, matrixSize, cudaMemcpyDeviceToHost);


    // Print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f microseconds\n", (milliseconds*1000));

    //
    string txt1 = "Transpose Matrix of size " + to_string(cols) + "*" + to_string(rows) + " stored as output_4_CS25MTECH12017.csv";
    string txt2 = "Kernel execution time: " + to_string(milliseconds*1000) + " microseconds";
    writeMatixToCSVfile("public_test_cases/output_4_CS25MTECH12017.csv", matrixTranspose, cols, rows);
    writeTXTFile("public_test_cases/output_4_CS25MTECH12017.txt", txt1, txt2);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //deallocating GPU mem
    cudaFree(d_matrix);
    cudaFree(d_matrixTranspose);

    //deallocating CPU mem
    free(matrix);
    free(matrixTranspose);

    return 0;
}