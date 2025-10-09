#include <iostream>
#include <vector>
#include <string>

using namespace std;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error at " << file << ":" << line << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}