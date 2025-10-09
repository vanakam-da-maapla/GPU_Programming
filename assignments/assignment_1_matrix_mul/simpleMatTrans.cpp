#include <iostream>
#include <vector>
#include <string>
#include "fileoperations.h"

using namespace std;


// Function to transpose a matrix represented as a 1D array
void transposeMatrix(int* original, int* transposed, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int original_index = i * cols + j;
            int transposed_index = j * rows + i;
            transposed[transposed_index] = original[original_index];
        }
    }
}

// Helper function to print a 1D matrix
void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    
    int *matrix_a, rows, cols;
    setMatrixFromFile(false, "public_test_cases/matrix_a.csv", matrix_a, rows, cols);

    // Allocate memory for the transposed matrix
    int* transposed_matrix = new int[cols * rows];

    // Perform the transpose
    transposeMatrix(matrix_a, transposed_matrix, rows, cols);

    // Print the results

    writeMatixToCSVfile("output_matrixTrans_cpu.csv", transposed_matrix, cols, rows);

    // Clean up dynamically allocated memory
    delete[] matrix_a;
    delete[] transposed_matrix;

    return 0;
}