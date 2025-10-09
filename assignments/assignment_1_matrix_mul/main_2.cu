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

compile: nvcc -arch=sm_80 main_2.cu fileoperations.cpp -o matmul_tiled
run: ./matmul_tiled 32 "public_test_cases/matrix_a.csv" "public_test_cases/matrix_b.csv"

************************************************************************************************/


__global__ void matrixMulTiled(int *matrix_a, int *matrix_b, int *matrix_c, const unsigned rows_a, const unsigned cols_a, const unsigned rows_b, const unsigned  cols_b, const unsigned  tileWidth){
    int accThreadNum_x = (blockIdx.y * blockDim.y) + threadIdx.y;
    int accThreadNum_y = (blockIdx.x * blockDim.x) + threadIdx.x;


    //
    const unsigned tileDim = tileWidth * tileWidth;

    //
    extern __shared__ int sharedMem[];
    int *s_matrix_a = sharedMem;
    int *s_matrix_b = &sharedMem[tileDim];
    int *s_matrix_c = &sharedMem[2*tileDim];
    // extern __shared__ int *s_matrix_c;

    s_matrix_c[(threadIdx.y * tileWidth) + threadIdx.x] = 0;

    // printf("\nmatrixMulTiled debug");
    int tileIndex;
    for(tileIndex=0; tileIndex < (cols_a+tileWidth); tileIndex += tileWidth){
        const unsigned tileIndexMatrix_a = (accThreadNum_x * cols_a) + ((accThreadNum_y % tileWidth) + tileIndex);
        const unsigned tileIndexMatrix_b = ((tileIndex + (accThreadNum_x % tileWidth)) * cols_b) + (accThreadNum_y);
        
        if(tileIndexMatrix_a < (rows_a * cols_a)){
            // printf("\ntileIndexMatrix_a, %d", tileIndexMatrix_a);
            s_matrix_a[(threadIdx.y * tileWidth) + threadIdx.x] = matrix_a[tileIndexMatrix_a];
        } else s_matrix_a[(threadIdx.y * tileWidth) + threadIdx.x] = 0;
        if(tileIndexMatrix_b < (rows_b * cols_b)){
            // printf("\ntileIndexMatrix_b, %d", tileIndexMatrix_b);
            s_matrix_b[(threadIdx.y * tileWidth) + threadIdx.x] = matrix_b[tileIndexMatrix_b];
        } else s_matrix_b[(threadIdx.y * tileWidth) + threadIdx.x] = 0;
        __syncthreads();

        int sum=0;
        for(int commonPointer=0; commonPointer < tileWidth; commonPointer++){
            sum += s_matrix_a[(threadIdx.y * tileWidth) + commonPointer] * s_matrix_b[(commonPointer * tileWidth) + threadIdx.x];
        }

        s_matrix_c[(threadIdx.y * tileWidth) + threadIdx.x] += sum;
        __syncthreads();
    }

    if(accThreadNum_x < rows_a && accThreadNum_y < cols_b){
        matrix_c[(accThreadNum_x * cols_b) + accThreadNum_y] = s_matrix_c[(threadIdx.y* tileWidth) + threadIdx.x];
    }
    // printf("\nDevice debug C co-ord: %d, %d, %d", (accThreadNum_x * cols_b) , accThreadNum_y, tileIndex);


}


int main(int argc, char* argv[]){

    int tileWidth = stoi(argv[1]);   // #of thread blocks
    string mat_a_filename = argv[2];
    string mat_b_filename = argv[3];

    
    cout<<"\nFilename: "<<mat_a_filename <<" "<<mat_b_filename;

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //get matrix A&B from csv file
    int *matrix_a, *matrix_b, rows_a, cols_a, rows_b, cols_b;
    setMatrixFromFile(false, mat_a_filename, matrix_a, rows_a, cols_a);
    setMatrixFromFile(false, mat_b_filename, matrix_b, rows_b, cols_b);

     //declare dimensions of a  blocks and threads 
    dim3 gridDim3(ceil(static_cast<double>(rows_a)/tileWidth), ceil(static_cast<double>(cols_b)/tileWidth));
    dim3 blockDim3(tileWidth, tileWidth);  

    printf("Host debug 2: %f, %f", ceil(static_cast<double>(rows_a)/tileWidth), ceil(static_cast<double>(cols_b)/tileWidth));

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

    cout<<"\nHost debug 3";

    //copy matrix A&B host to GPU mem
    cudaMemcpy(d_matrix_a, matrix_a, matrixSize_a, cudaMemcpyHostToDevice);
    cout<<"\nHost debug 4";
    cudaMemcpy(d_matrix_b, matrix_b, matrixSize_b, cudaMemcpyHostToDevice);

    //kernel launch
    cout<<"\nKernel launched...";
    cudaEventRecord(start);
    matrixMulTiled<<<gridDim3, blockDim3, (3 * tileWidth * tileWidth * sizeof(int))>>>(d_matrix_a, d_matrix_b, d_matrix_c, rows_a, cols_a, rows_b, cols_b, tileWidth);
    cudaEventRecord(stop);

    //copy matrix C from device to host
    cudaMemcpy(matrix_c, d_matrix_c, matrixSize_c, cudaMemcpyDeviceToHost);

    //write the matrix into output CSV file
    writeMatixToCSVfile("public_test_cases/output_2_CS25MTECH12017.csv", matrix_c, rows_a, cols_b);

    // Print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f micro-seconds\n", (milliseconds * 1000));

    //write the matrix into output CSV file
    string txt1 = "Product Matrix of size " + to_string(rows_a) + "*" + to_string(cols_b) + " stored as output_2_CS25MTECH12017.csv";
    string txt2 = "Kernel execution time: " + to_string(milliseconds*1000) + " microseconds";
    writeMatixToCSVfile("public_test_cases/output_2_CS25MTECH12017.csv", matrix_c, rows_a, cols_b);
    writeTXTFile("public_test_cases/output_2_CS25MTECH12017.txt", txt1, txt2);


    // free mem in GPU
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    // free mem in CPU
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);


    return 0;
}