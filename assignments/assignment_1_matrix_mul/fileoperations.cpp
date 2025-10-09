#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "fileoperations.h"

using namespace std;


void setMatrixFromFile(bool print, string filename, int *&finalMatrix, int &rows, int &cols){

    vector<vector<int>> matrix;
    ifstream matrixFile(filename);

    if(!matrixFile.is_open()){
        cout<<"Not able to open file!...";
    }

    cout<<"\nEntered setMatrixFromFile....";

    string line;
    while(getline(matrixFile, line)){
        vector<int> row;
        stringstream ss(line);
        string cell;

        while(getline(ss, cell, ',')){
            row.push_back(stoi(cell));
        }
        matrix.push_back(row);
    }

    matrixFile.close();
    
    rows = matrix.size();
    cols = rows > 0 ? matrix[0].size() : 0;

    //allocate memory
    finalMatrix = (int*) malloc(rows * cols * sizeof(int*));

    //stroing 2-d vector into 1-d array 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            finalMatrix[(i * cols) + j] = matrix[i][j];
        }
    }

    //printing it
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         cout<< finalMatrix[(i * cols) + j] << ", ";
    //     }
    //     cout<< endl;
    // }

    cout<< "\nfinalMatrix size: " << sizeof(finalMatrix) << " " << rows;

    cout<<"\nSuccessfully matrix copied from filename: "<< filename <<endl;

}

void writeMatixToCSVfile(string filename, int *&matrix, int rows, int cols){

    ofstream csvFile(filename);

    if(!csvFile.is_open()){
        cout<<"\nNot able to open CSV file";
    }
    // cout<<"\nrowSize: " << rowSize << " " << matrix[0];
    for(size_t i=0; i<rows; i++){

        for(int j=0; j<cols; j++){
             csvFile << matrix[(i * cols) + j]; // Write the integer value

            // Add a comma after each element, except the last one in a row
            if (j < cols - 1) {
                csvFile << ",";
            }
        }
        csvFile << "\n";
    }

    csvFile.close();

    cout<<"\nSuccessfully matrix copied to file: "<< filename <<endl;


}


void writeTXTFile(string filename, string text1, string text2){

    ofstream csvFile(filename);

    if(!csvFile.is_open()){
        cout<<"\nNot able to open CSV file";
    }

    csvFile << text1 << endl;
    csvFile << text2 << endl;

    csvFile.close();

    cout<<"\nSuccessfully txt copied to file: "<< filename <<endl;


}
