#include "vector.h"


using namespace std;

 Vector::~Vector(){

    delete v;

}

 int Vector::GetSizeVector() {

     return n;

 }


 Vector::Vector(int size){

    this->n = size;

    int i;

    this->v = (float*) malloc(sizeof(float)*this->n);

    for(i=0; i<this->n;i++){
        this->v[i] = 0;
    }

}

 Vector::Vector(char *filename){

    int i;

    ifstream file(filename);


    if(!file.is_open()){
        cout << "Nao foi possivel abrir o arquivo!" << endl;
        exit(2);
    }

    file >> this->n;

    this->v = (float*) malloc(sizeof(float)*this->n);

    for(i=0; i<this->n;i++){
        file >> this->v[i];
    }

    file.close();

}


float* Vector::GetDataVector(){

    return v;

 }

float Vector::GetElementVector(int i){

    return v[i];

}

float Vector::operator[](int i){

    return v[i];

}

void Vector::ResetVector(){

    memset(v, 0, sizeof(float) * n);

}

void Vector::PrintScreenVector(){

    int i;

    for(i=0;i<this->n;i++)
        cout << this->v[i] << " ";

    cout << endl;

}

void Vector::PrintFileVector(char *filename){


    ofstream file;
    int i;

    file.open(filename);

    for(i=0;i<n;i++)
        file << v[i] << " ";

    file << "\n";

    file.close();

}

Vector *Vector::SelectElements(vector<int> *selectedElems) {

        Vector *sv;
        int i;

        sv = new Vector((int)selectedElems->size());

        for (i = 0; i < sv->GetSizeVector(); i++) {
            sv->SetElementVector(i, v[selectedElems->at(i)]);
        }

        return sv;

}

void Vector::SetElementVector(int i, float element){

    v[i] = element;

}

/* Copia a estrutura */
Vector *Vector::Copy() {

    Vector *v;
    float *aux;

    v = new Vector(n);
    aux = v->GetDataVector();
    memcpy(aux, this->GetDataVector(), sizeof(float) * n);

    return v;

}
