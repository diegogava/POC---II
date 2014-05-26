#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>

#include <stdio.h>
#include <vector>

using namespace std;

class Vector {

public:

    /* Atributos */
    float* v; /* Vetor de ponto flutuante. */
    int n; /* Numero de elementos do vetor. */

    /* Metodos */
    Vector(int size);
    Vector(char* filename);
    ~Vector();
    Vector* Copy();

    void ResetVector();
    void SetElementVector(int i, float element);
    void PrintScreenVector();
    void PrintFileVector(char *filename);

    int	 GetSizeVector();

    float GetElementVector(int i);
    float* GetDataVector();
    float operator[](int i);

    Vector *SelectElements(vector<int> *selectedElems);

};


#endif // VECTOR_H
