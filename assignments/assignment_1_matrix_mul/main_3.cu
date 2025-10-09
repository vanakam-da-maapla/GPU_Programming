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

compile: nvcc -arch=sm_80 main_3.cu fileoperations.cpp -o mattrans_basic
run: ./mattrans_basic 1024 1024 "public_test_cases/matrix_a.csv"

************************************************************************************************/

__global__ void matTranspose(int *matrix, int *matrixTranspose, int rows, int cols){
    int accThreadNum = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(accThreadNum < rows * cols){
        int currRowNum = accThreadNum / cols;
        int currColNum = accThreadNum % cols;
        matrixTranspose[(currColNum * rows) + currRowNum] = matrix[(currRowNum * cols) + currColNum];
    }
}


int main(int argc, char* argv[]){

    int cudaGridSize = stoi(argv[1]);   // #of thread blocks 
    int cudaBlockSize = stoi(argv[2]);  // #of threads in a block
    string matrixFilename = argv[3];

    
    cout<<"\nFilename: "<<matrixFilename;

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //get matrix A&B from csv file
    int *matrix, rows, cols;
    setMatrixFromFile(false, matrixFilename, matrix, rows, cols);

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
    matTranspose<<<cudaGridSize, cudaBlockSize>>>(d_matrix, d_matrixTranspose, rows, cols);
    cudaEventRecord(stop);

    //
    cudaMemcpy(matrixTranspose, d_matrixTranspose, matrixSize, cudaMemcpyDeviceToHost);

    // Print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f microseconds\n", (milliseconds*1000));

    //
    string txt1 = "Transpose Matrix of size " + to_string(cols) + "*" + to_string(rows) + " stored as output_3_CS25MTECH12017.csv";
    string txt2 = "Kernel execution time: " + to_string(milliseconds*1000) + " microseconds";
    writeMatixToCSVfile("public_test_cases/output_3_CS25MTECH12017.csv", matrixTranspose, cols, rows);
    writeTXTFile("public_test_cases/output_3_CS25MTECH12017.txt", txt1, txt2);

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