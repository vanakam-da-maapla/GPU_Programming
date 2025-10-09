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

compile: nvcc -arch=sm_80 main_1_2.cu fileoperations.cpp -o matmul_2d
run: ./matmul_2d 4 4 16 16 "public_test_cases/matrix_a.csv" "public_test_cases/matrix_b.csv"

************************************************************************************************/

__global__ void matrixMul2D(int *matrix_a, int *matrix_b, int *matrix_c, int rows_a, int cols_a, int rows_b, int cols_b){
    int accThreadNum_x = (blockIdx.y * blockDim.y) + threadIdx.y; // actuall thread number in x-dir
    int accThreadNum_y = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(accThreadNum_x < rows_a && accThreadNum_y < cols_b){
        int sum=0;
        for(int commonPointer=0; commonPointer<cols_a; commonPointer++){
            // printf("\nmatrixMul2D 4 %d, %d, %d, %d:, %d ", accThreadNum_x, accThreadNum_y, commonPointer, matrix_a[((accThreadNum_x * cols_a) + commonPointer)], matrix_b[((commonPointer * cols_b) + accThreadNum_y)]);
            sum += matrix_a[(accThreadNum_x * cols_a) + commonPointer] * matrix_b[(commonPointer * cols_b) + accThreadNum_y];
            // printf("\nmatrixMul2D 5");
        }
        matrix_c[(accThreadNum_x * cols_b) + accThreadNum_y] = sum;
        // printf("\nmatrixMul2D: %d, %d, %d", accThreadNum_x, accThreadNum_y, sum);
    } 
}


int main(int argc, char* argv[]){

    //
    int cudaGridDim_x = stoi(argv[1]);   // #of thread blocks 
    int cudaGridDim_y = stoi(argv[2]); 
    int cudaBlockDim_x = stoi(argv[3]);  // #of threads in a block
    int cudaBlockDim_y = stoi(argv[4]); 
    string mat_a_filename = argv[5];
    string mat_b_filename = argv[6];

    cout<<"\nFilename: "<<mat_a_filename <<" "<<mat_b_filename;

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //declare dimensions of a  blocks and threads 
    dim3 gridDim3(cudaGridDim_x, cudaGridDim_y);
    dim3 blockDim3(cudaBlockDim_x, cudaBlockDim_y);

    //get matrix A&B from csv file
    int *matrix_a, *matrix_b, rows_a, cols_a, rows_b, cols_b;
    setMatrixFromFile(false, mat_a_filename, matrix_a, rows_a, cols_a);
    
    //printing it
    // for (int i = 0; i < rows_a; ++i) {
    //     for (int j = 0; j < cols_a; ++j) {
    //         cout<< matrix_a[(i * cols_a) + j] << ", ";
    //     }
    //     cout<< endl;
    // }

    setMatrixFromFile(false, mat_b_filename, matrix_b, rows_b, cols_b);

    //Declare the output matrix C
    int matrixSize_a = rows_a * cols_a * sizeof(int);
    int matrixSize_b = rows_b * cols_b * sizeof(int);
    int matrixSize_c = rows_a * cols_b * sizeof(int);

    int *matrix_c = (int*) malloc(matrixSize_c);
    int *d_matrix_a, *d_matrix_b, *d_matrix_c;

    //allocate mem in GPU
    cudaMalloc(&d_matrix_a, matrixSize_a);
    cudaMalloc(&d_matrix_b, matrixSize_b);
    cudaMalloc(&d_matrix_c, matrixSize_c);

    //copy matrix A&B host to GPU mem
    cudaMemcpy(d_matrix_a, matrix_a, matrixSize_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, matrixSize_b, cudaMemcpyHostToDevice);

    //launch the kernal with custom dim 
    cudaEventRecord(start);
    matrixMul2D<<<gridDim3, blockDim3>>>(d_matrix_a, d_matrix_b, d_matrix_c, rows_a, cols_a, rows_b, cols_b);
    cudaEventRecord(stop);

    //blocking main thread until all GPU operations completes before this line(optional since we using cudaMemcpy in next immediate line)
    cudaEventSynchronize(stop);

    //copy computed matrix C from device(GPU) to host(CPU)
    cudaMemcpy(matrix_c, d_matrix_c, matrixSize_c, cudaMemcpyDeviceToHost);

    // Print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f micro-seconds\n", (milliseconds*1000));

    //write the matrix into output CSV file
    string txt1 = "Product Matrix of size " + to_string(rows_a) + "*" + to_string(cols_b) + " stored as output_1_2_CS25MTECH12017.csv";
    string txt2 = "Kernel execution time: " + to_string(milliseconds*1000) + " microseconds";
    writeMatixToCSVfile("public_test_cases/output_1_2_CS25MTECH12017.csv", matrix_c, rows_a, cols_b);
    writeTXTFile("public_test_cases/output_1_2_CS25MTECH12017.txt", txt1, txt2);

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