/*
 * The implementation of the NIPALS algorithm provided in this library
 * is a translation from the MATLAB version of the NIPALS algorithm
 * written by Dr. Herve Abdi from The University of Texas at Dallas
 * http://www.utdallas.edu/~herve.
 */

#ifndef PLS_H
#define PLS_H

#define EPSILON 0.000001

#include "vector.h"
#include "matrix.h"

using namespace std;

class PLS {

public:

    Vector *Xmean, *Xstd, *Yorig, *b;
    Matrix *T, *P, *W, *Wstar;
    int maxFactors;		// maximum number of factors for this model

    // other variables
    Vector *zdataV;		// variable to hold result of zscore


    void normaliz(Vector *v, Vector *retvector);
    void MultiplyTransposeMatrixbyVector(Matrix *M, Vector *v, Vector *retvector);
    void MultiplyMatrixbyVector(Matrix *M, Vector *v, Vector *retvector);
    float MultiplyVectorTransposedbyVector(Vector *v1, Vector *v2);
    void MultiplyVectorandScalar(Vector *v, float s, Vector *retvector);
    void SubtractFromMatrix(Matrix *M, Vector *t, Vector *p);
    void SubtractFromVector(Vector *v, Vector *t, float c, float bl);
    void CopyVector(Vector *v, Vector *retvector);
    void mean(Matrix *M, Vector *retvector) ;
    void mean(Vector *v, Vector *retvector) ;
    void std(Matrix *M, Vector *mean, Vector *retvector);
    void std(Vector *v, Vector *mean, Vector *retvector);
    void zscore(Matrix *M, Vector *mean, Vector *std);
    void zscore(Vector *v, Vector *mean, Vector *std);
    void ComputeWstar();

    void CreatePLSModel(Matrix *X, Vector *Y, int nfactors);

    // clear data not used for PLS regression (leave only Bstar, Ymean, Xstd, Xmean)
    void ClearExtraMatricesPLSReg();

    // set all matrices initializing with already computed values
    // Warning: this function COPY all variables
    void InitializePLSModel(Vector *Xmean, Vector *Xstd, Vector *Yorig, Vector *b, Matrix *T,
            Matrix *P, Matrix *W, Matrix *Wstar);

    // remove matrices not used for projections
    void ClearExtraMatrices();

    // return projection considering n factors
    void Projection(Vector *feat, Vector *retproj, int nfactors);

    // execute PLS for maximum number of factors: nfactor
    void runpls(Matrix *X, Vector *Y, int nfactor, char *OutputDir = NULL, float ExplainedX = 1, float ExplainedY = 1);

    // return feature vector running zcore
    void ExecuteZScore(float *feat, float *zscoreResult);

    // set matrices for PLS
    void SetMatrices(Matrix *W, Matrix *Wstar, Matrix *P, Vector *Xmean, Vector *Xstd,Vector *b);


    // clear variables of this class
    void ClearPLS();


    // Return matrices and vectors
    Matrix *GetWMatrix() { return W; }
    Matrix *GetTMatrix() { return T; }
    Matrix *GetPMatrix() { return P; }
    Matrix *GetWstar() { return Wstar; }
    Vector *GetbVector() { return b; }
    Vector *GetYVector() { return Yorig; }
    Vector *GetMeanVector() { return Xmean; }
    Vector *GetStdVector() { return Xstd; }
    Vector *GetBstar(int nfactors);



public:
    PLS();
    ~PLS();
};


#endif
