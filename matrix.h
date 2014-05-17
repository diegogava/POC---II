#ifndef MATRIZ_H
#define MATRIZ_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <string.h>
#include <malloc.h>
#include <emmintrin.h>
#include <vector>
#include <cstring>

using namespace std;

class Matrix {

public:


/* Metodos */
Matrix(int r,int c);
Matrix(char *filename);
~Matrix();

int	 GetNumberRows();
int	 GetNumberColumns();

float GetElementMatrix(int i,int j);
float* GetDataMatrix();
float* GetColumn(int index);

void SetElementMatrix(int i, int j, float element);
void PrintScreenMatrix();
void PrintFileMatrix(char* filename);

Matrix* CopyMatrix();
Matrix* ConcatenataLinhasMatrizes(Matrix *m1, Matrix *m2);
Matrix* GetSelectedCols(vector<int> *selectedcols);

/* Atributos */
float* m; /* Vetor de ponto flutuante, que funcionara como sendo a matriz de ponto flutuante. */
int rows; /* Numero de linhas da matriz. */
int cols; /* Numero de colunas da matriz. */

};


/* Multiplica duas matrizes. */
Matrix *MultiplyMatrices(Matrix *m1, Matrix *m2);

/* Calcula a matriz inversa. */
Matrix* MatrixInverse(Matrix *matrix);

/* Calcula a matriz transposta. */
Matrix *MatrixTransposed(Matrix *m);

// dot product when n is not multiple of 4
float DotProductSSENotMultof4(float *v, float *v2, int n);

void ZscoreSSE(float *data, float *mean, float *std, float *outputvect, int n);

#endif //MATRIZ_H
