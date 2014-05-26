#include "matrix.h"
#include <stdlib.h>

using namespace std;

/* Construtor que recebe por parâmetro a quantidade de linhas e de colunas da matriz, e instancia um objeto do tipo Matrix. */
Matrix::Matrix(int r,int c){

    rows = r;
    cols = c;
    int i;

     m = (float*)malloc(sizeof(float)*(rows*cols));

     for(i = 0;i < rows*cols;i++)
         m[i] = 0;


}

/* Construtor que recebe por parâmetro um arquivo com a quantidade de linhas e de colunas da matriz na primeira linha, e nas linhas posteriores os valores (em ponto flutuante) dos elementos, e instancia um objeto do tipo Matrix. */
Matrix::Matrix(char* filename){

   int i,j;

   ifstream file(filename);

   if(!file.is_open()){
       cout << "Nao foi possivel abrir o arquivo!" << endl;
       exit(2);
   }

   file >> this->rows >> this->cols;

   this->m = (float*) malloc(sizeof(float)*(this->rows*this->cols));

   for(i=0;i<this->rows; i++){

       for(j=0;j<this->cols;j++)

           file >> this->m[j*this->rows +i];

   }

   file.close();

}

/* Destrutor */
Matrix::~Matrix(){

    delete this->m;

}

/* Retorna o número de linhas da matriz. */
int Matrix::GetNumberRows() {

    return this->rows;

}

/* Retorna o número de colunas da matriz. */
int Matrix::GetNumberColumns() {

    return this->cols;

}

/* Metodo que retorna o elemento da matriz. */
float Matrix::GetElementMatrix(int i, int j){

    return this->m[j*this->rows+i];

}

/* Seta o valor de ponto flutuante passado por parametro na linha i e coluna j. */
void Matrix::SetElementMatrix(int i, int j, float element){

    this->m[j*this->rows+i] = element;

}

/* Retorna o ponteiro de ponto flutuante com os dados (elementos de ponto flutuante) da matriz. */
float* Matrix::GetDataMatrix(){

    return this->m;

}

/* Obtem a coluna, atraves do seu indice, retornando o ponteiro correspondente. */
float* Matrix::GetColumn(int index){

    return this->m + this->rows*index;

}

/* Imprime a matriz na tela. */
void Matrix::PrintScreenMatrix(){

    int i,j;

    for(i=0;i<this->rows;i++){

        for(j=0;j<this->cols*this->rows;j+=this->cols){

            cout << this->m[i+j] << " ";

        }

        cout << "\n";

    }

}

/* Imprime a matriz em um arquivo. */
void Matrix::PrintFileMatrix(char *filename){

    ofstream file;
    int i,j;

    file.open(filename);

    file << this->rows << " " << this->cols << "\n";

    for(i=0;i<rows;i++){

        for(j=0;j<this->cols*this->rows;j+=this->cols){

            file << this->m[i+j] << " ";

        }

        file << "\n";

    }

    file.close();

}

/* Copia os dados da matriz */
Matrix* Matrix::CopyMatrix() {

    Matrix *m1;
    float *data;

    m1 = new Matrix(this->rows, this->cols);
    data = m1->GetDataMatrix();
    memcpy(data, this->GetDataMatrix(), sizeof(float) * this->rows * this->cols);

    return m1;

}

/* Concatena as linhas das matrizes, primeiro m1, depois m2. */
Matrix *Matrix::ConcatenataLinhasMatrizes(Matrix *m1, Matrix *m2) {

    Matrix *m_aux;
    int i, j, idxrow;

        /* Copia somente a segunda matriz. */
        if (m1 == NULL) {
            m_aux = m2->CopyMatrix();
            return m_aux;
        }

        /* Testa a Compatibilidade */
        if (m1->GetNumberColumns() != m2->GetNumberColumns()) {
            printf("Numero incompativel de colunas!\n");
            exit(2);
        }

        m_aux = new Matrix (m1->GetNumberRows() + m2->GetNumberRows(), m1->GetNumberColumns());

        idxrow = 0;
        // m1
        for (i = 0; i < m1->GetNumberRows(); i++) {
            for (j = 0; j < m1->GetNumberColumns(); j++) {
                m_aux->SetElementMatrix(j, idxrow, m1->GetElementMatrix(j, i));
            }
            idxrow++;
        }
        // m2
        for (i = 0; i < m2->GetNumberRows(); i++) {
            for (j = 0; j < m2->GetNumberColumns(); j++) {
                m_aux->SetElementMatrix(j, idxrow, m2->GetElementMatrix(j, i));
            }
            idxrow++;
        }

        return m_aux;
}

Matrix* Matrix::GetSelectedCols(vector<int> *selectedcols) {

    Matrix *matrix;
    int i, j, k;


    matrix = new Matrix (this->GetNumberRows(), (int) selectedcols->size());

    for (i = 0; i < (int)selectedcols->size(); i++) {
        j = selectedcols->at(i);
        for (k = 0; k < this->GetNumberRows(); k++) {
            matrix->SetElementMatrix(i, k, this->GetElementMatrix(j, k));
        }
    }

    return matrix;
}



/*Calcula a matriz inversa.Utilizando o metodo de Gauss-Jordan */
Matrix* MatrixInverse(Matrix *matrix){

    Matrix* inverse;

    float pivo,mult;
    int i,j,k;

    if(matrix->GetNumberRows() != matrix->GetNumberColumns()){
        cout << "Nao eh possivel calcular a inversa, pois a matriz nao eh quadrada!\n";
        exit(2);
    }

    inverse = new Matrix(matrix->GetNumberRows(),matrix->GetNumberColumns());

    /* Definindo a Matriz Identidade */

    for(i=0;i<matrix->GetNumberRows();i++){
            for(j=0;j<matrix->GetNumberRows();j++){
                if(i == j){
                    inverse->SetElementMatrix(i,j,1);
                }
                else{
                    inverse->SetElementMatrix(i,j,0);
                }
            }
        }


    /* Calculando Inversa */
    for(j = 0; j < matrix->GetNumberRows(); j++){

        pivo = matrix->GetElementMatrix(j,j); /* Elementos da diagonal principal. */

        for(k = j; k < matrix->GetNumberRows(); k++){

            matrix->SetElementMatrix(j,k, (matrix->GetElementMatrix(j,k))/(pivo)); /* L1 -> L1/pivo , L2 -> L2/pivo, L3 -> L3/pivo */
            inverse->SetElementMatrix(j,k,(inverse->GetElementMatrix(j,k)/(pivo))); /* Ex: 1 0 0 / pivo  , 0 1 0 / pivo,   0 0 1/ pivo */

        }

        for(i = 0; i < matrix->GetNumberRows(); i++){

            if(i != j){

                mult = matrix->GetElementMatrix(i,j);/* Multiplicador */

                for(k = 0; k < matrix->GetNumberRows(); k++){

                    matrix->SetElementMatrix(i,k, matrix->GetElementMatrix(i,k) - (mult * matrix->GetElementMatrix(j,k)) ) ; /* Ex: L2 -> L2 - ( 1"m" - L1) */
                    inverse->SetElementMatrix(i,k, (inverse->GetElementMatrix(i,k) - (mult * inverse->GetElementMatrix(j,k))));

                }
            }
        }
    }


    return inverse;

}

/* Calcula a matriz transposta. */
Matrix* MatrixTransposed(Matrix *matrix){

    Matrix *transposed;
    int i,j;

    transposed = new Matrix(matrix->GetNumberColumns(),matrix->GetNumberRows());

    for(i=0;i<matrix->GetNumberColumns();i++){

        for(j=0;j<matrix->GetNumberRows();j++){

            transposed->SetElementMatrix(i,j, matrix->GetElementMatrix(j,i));

        }

    }

    return transposed;

}


float DotProductSSENotMultof4(float *v, float *v2, int n) {

    __m128 a, b, c, d;
    float out_sse[4];
    int i, n2;

    d = _mm_setzero_ps();
    if ((n & 3) == 0) { // multiple of 4

        for (i = 0; i < n; i += 4) {
            a = _mm_loadu_ps(v + i);
            b = _mm_loadu_ps(v2 + i);
            c = _mm_mul_ps(a, b);
            d = _mm_add_ps(c, d);
        }
        _mm_storeu_ps(out_sse,d);
    }
    else {  // n not multiple of 4
        n2 = n - 4;
        for (i = 0; i < n2; i += 4) {
            a = _mm_loadu_ps(v + i);
            b = _mm_loadu_ps(v2 + i);
            c = _mm_mul_ps(a, b);
            d = _mm_add_ps(c, d);
        }
        _mm_storeu_ps(out_sse,d);
        n2 = n - (n & 0x3);
        // do the remaining elements
        for (i = n2; i < n; i++) {
            out_sse[0] += v[i] * v2[i];
        }
    }

    return out_sse[0] + out_sse[1] + out_sse[2] + out_sse[3];
}


void ZscoreSSE(float *data, float *mean, float *std, float *outputvect, int n) {

__m128 a, b, c, d, e;
int i, n2;

    if ((n & 3) == 0) { // multiple of 4
        for (i = 0; i < n; i += 4) {
            a = _mm_loadu_ps(data + i);
            b = _mm_loadu_ps(mean + i);
            c = _mm_loadu_ps(std + i);
            d = _mm_sub_ps(a, b);
            e = _mm_div_ps(d, c);
            _mm_storeu_ps(outputvect + i, e);
        }
    }
    else {  // n not multiple of 4
        n2 = n - 4;
        for (i = 0; i < n2; i += 4) {
            a = _mm_loadu_ps(data + i);
            b = _mm_loadu_ps(mean + i);
            c = _mm_loadu_ps(std + i);
            d = _mm_sub_ps(a, b);
            e = _mm_div_ps(d, c);
            _mm_storeu_ps(outputvect + i, e);
        }
        // do the remaining elements
        for (i = n2; i < n; i++) {
            outputvect[i] = (data[i] - mean[i]) / std[i];
        }
    }
}
