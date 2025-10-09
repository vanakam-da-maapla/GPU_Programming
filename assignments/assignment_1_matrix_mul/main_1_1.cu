#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fileoperations.h"
#include "cudaExceptionHandler.cu"

using namespace std;

/************************************************************************************************
                        COMMAND TO RUN THIS RPOGRAM

compile: nvcc -arch=sm_80 main_1_1.cu fileoperations.cpp -o matmul_1d
run: ./matmul_1d 4 1024 "public_test_cases/matrix_a.csv" "public_test_cases/matrix_b.csv"

************************************************************************************************/

__global__ void matrixMul(int *matrix_a, int *matrix_b, int *matrix_c, int rows_a, int cols_a, int rows_b, int cols_b){
    int accThreadNum = (blockIdx.x * blockDim.x) + threadIdx.x;
    int currRowNum = accThreadNum / cols_b;
    int currColNum = accThreadNum % cols_b;
    if(currRowNum < rows_a && currColNum < cols_b){
        //printf("accThreadNum: %d, %d, %d", accThreadNum,currRowNum, currColNum);
        int sum = 0;
        int commonPointer=0;
        while(commonPointer < cols_a){
            // printf("commonPointer: %d, %d, %d, %d, %d, %d, %d \n: ", commonPointer ,currRowNum, currColNum, ((currRowNum * eachDimSize) + commonPointer), ((eachDimSize * commonPointer) + currColNum), matrix_a[(currRowNum * eachDimSize) + commonPointer], matrix_b[(eachDimSize * commonPointer) + currColNum]);
            sum += matrix_a[(currRowNum * cols_a) + commonPointer] * matrix_b[(cols_b * commonPointer) + currColNum];
            commonPointer++;
        }
        matrix_c[(currRowNum * cols_b) + currColNum] = sum;
    }
}

int main(int argc, char* argv[]){

    int cudaGridSize = stoi(argv[1]);   // #of thread blocks 
    int cudaBlockSize = stoi(argv[2]);  // #of threads in a block
    string mat_a_filename = argv[3];
    string mat_b_filename = argv[4];

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //
    int *matrix_a, *matrix_b, rows_a, cols_a, rows_b, cols_b;
    setMatrixFromFile(false, mat_a_filename, matrix_a, rows_a, cols_a);
    setMatrixFromFile(false, mat_b_filename, matrix_b, rows_b, cols_b);
    
    // print matrix
    //  for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         cout<< matrix_b[(i*rows) + j] << ", ";
    //     }
    //     cout<< endl;
    // }

    //
    unsigned int matrixSize_a = rows_a * cols_a * sizeof(int);
    unsigned int matrixSize_b = rows_b * cols_b * sizeof(int);
    unsigned int matrixSize_c = rows_a * cols_b * sizeof(int);
    int *matrix_c = (int*) malloc(matrixSize_c);
    int *d_matrix_a, *d_matrix_b, *d_matrix_c;

    //
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a, matrixSize_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b, matrixSize_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c, matrixSize_c));

    //
    cudaMemcpy(d_matrix_a, matrix_a, matrixSize_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, matrixSize_b, cudaMemcpyHostToDevice);

    //
    cudaEventRecord(start);
    matrixMul<<<cudaGridSize, cudaBlockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, rows_a, cols_a, rows_b, cols_b);
    cudaEventRecord(stop);

    //
    cudaMemcpy(matrix_c, d_matrix_c, matrixSize_c, cudaMemcpyDeviceToHost);

    // Print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f micro-seconds\n", (milliseconds*1000));

    //
    string txt1 = "Product Matrix of size " + to_string(rows_a) + "*" + to_string(cols_b) + " stored as output_1_1_CS25MTECH12017.csv";
    string txt2 = "Kernel execution time: " + to_string(milliseconds*1000) + " microseconds";
    writeMatixToCSVfile("public_test_cases/output_1_1_CS25MTECH12017.csv", matrix_c, rows_a, cols_b);
    writeTXTFile("public_test_cases/output_1_1_CS25MTECH12017.txt", txt1, txt2);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //deallocating GPU mem
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    //deallocating CPU mem
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);


    return 0;
}