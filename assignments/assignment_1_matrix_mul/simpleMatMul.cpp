#include <iostream>
#include <vector>
#include <string>
#include "fileoperations.h"

using namespace std;


int main(int argc, char* argv[]){


    //get matrix A&B from csv file
    int *matrix_a, *matrix_b, rows_a, cols_a, rows_b, cols_b;
    setMatrixFromFile(false, "public_test_cases/matrix_a.csv", matrix_a, rows_a, cols_a);
    setMatrixFromFile(false, "public_test_cases/matrix_b.csv", matrix_b, rows_b, cols_b);

    //Declare the output matrix C
    int matrixSize_a = rows_a * cols_a * sizeof(int);
    int matrixSize_b = rows_b * cols_b * sizeof(int);
    int matrixSize_c = rows_a * cols_b * sizeof(int);
    int *matrix_c = (int*) malloc(matrixSize_c);


    cout<<"\nNormal multiplication:-"<<endl;
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            // Initialize the element of the result matrix to 0
            float sum = 0;
            
            for (int k = 0; k < cols_a; k++) {
                // Add the product of the corresponding elements to the sum
                sum += matrix_a[i * cols_a + k] * matrix_b[k * cols_b + j];
            }
            
            // After the inner loop, assign the final sum to the result matrix
            matrix_c[i * cols_b + j] = sum;
        }
    }

    //
    writeMatixToCSVfile("output_matrix_cpu.csv", matrix_c, rows_a, cols_b);

    // free mem in CPU
    free(matrix_a);
    free(matrix_b);


    return 0;
}
