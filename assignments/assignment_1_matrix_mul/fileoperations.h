#include <vector>
#include <string>

using namespace std;

#ifndef FILEOPERATIONS_H
#define FILEOPERATIONS_H

void setMatrixFromFile(bool print, string filename, int *&finalMatrix, int &rows, int &cols);
void writeMatixToCSVfile(string filename, int *&matrix, int rows, int cols);
void writeTXTFile(string filename, string text1, string text2);

#endif