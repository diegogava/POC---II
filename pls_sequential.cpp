/*
 * The implementation of the NIPALS algorithm provided in this library
 * is a translation from the MATLAB version of the NIPALS algorithm
 * written by Dr. Herve Abdi from The University of Texas at Dallas
 * http://www.utdallas.edu/~herve.
 */

#include "pls_sequential.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

void PLS::normaliz(Vector *v, Vector *retvector) {

    double sqr = 0;
    float  sqrr;
    int i;

    for (i = 0; i < v->GetSizeVector() ; i++) {
        sqr += (double) (v->GetElementVector(i) * v->GetElementVector(i));
    }

    sqrr = (float) sqrt(sqr);

    for (i = 0; i < v->GetSizeVector(); i++) {
        retvector->SetElementVector(i, v->GetElementVector(i)* (1/sqrr));
    }
}


// multiply transposed of MrxMc matrix M by vr vector v, result is Mc vector
void PLS::MultiplyTransposeMatrixbyVector(Matrix *M, Vector *v, Vector *retvector) {

    float *ptM, *ptv, *ptret;
    int i, j;

    ptv = v->GetDataVector();
    ptret = retvector->GetDataVector();

    for (i = 0; i < (int) M->GetNumberColumns(); i++) {

        ptM = M->CopyColumn(i);
        ptret[i] = 0;

        for (j = 0; j < M->GetNumberRows(); j++) {
            ptret[i] += ptM[j] * ptv[j];
        }
    }
}


// multiply MrxMc matrix M by vr vector v, result is Mr vector
void PLS::MultiplyMatrixbyVector(Matrix *M, Vector *v, Vector *retvector) {

    float *ptM, *ptv, *ptret;
    int i, j;

    ptv = v->GetDataVector();
    retvector->ResetVector();
    ptret = retvector->GetDataVector();

    for (i = 0; i < M->GetNumberColumns(); i++) {
        ptM = M->CopyColumn(i);
        for (j = 0; j < M->GetNumberRows(); j++) {
            ptret[j] += ptM[j] * ptv[i];
        }
    }
}


// Multiply vectors v1' and v2, returns a single number
float PLS::MultiplyVectorTransposedbyVector(Vector *v1, Vector *v2) {

    float ret = 0;
    int i;

    for (i = 0; i < v1->GetSizeVector() ; i++)  {
        ret += v1->GetElementVector(i) * v2->GetElementVector(i);
    }

    return ret;
}



// Multiply vector by scalar, returns a vector size rv
void PLS::MultiplyVectorandScalar(Vector *v, float s, Vector *retvector) {

    int i;

    for (i = 0; i < v->GetSizeVector(); i++)  {
        retvector->SetElementVector(i,v->GetElementVector(i) * s);
    }

}


// t is rx1 and p is cx1
void PLS::SubtractFromMatrix(Matrix *M, Vector *t, Vector *p) {

    int i, j;
    float *ptM, *ptt, *ptp;

    ptt = t->GetDataVector();
    ptp = p->GetDataVector();

    // Xres=Xres-t*p';

    for (j=0;j<M->GetNumberColumns();j++) {
        ptM = M->CopyColumn(j);
        for (i = 0; i < M->GetNumberRows(); i++) {
            ptM[i] = ptM[i] - (ptt[i] * ptp[j]);
        }
    }
}


void PLS::SubtractFromVector(Vector *v, Vector *t, float c, float bl) {

    int i;

    // Yres=Yres-(b(l)*(t*c'));
    for (i = 0; i < v->GetSizeVector(); i++) {
        v->SetElementVector(i,v->GetElementVector(i) - (bl * (t->GetElementVector(i) * c)));
    //	v[i] = v[i] - (bl * (t[i] * c[0]));
    }
}


void PLS::CopyVector(Vector *v, Vector *retvector) {

    memcpy(retvector->GetDataVector(), v->GetDataVector(), v->GetSizeVector() * sizeof(float));

}


void PLS::mean(Matrix *M, Vector *retvector) {

    float *ptM, *ptv;
    int i,j;

    retvector->ResetVector();
    ptv = retvector->GetDataVector();
    for (j = 0; j < M->GetNumberColumns(); j++) {
        ptM = M->CopyColumn(j);
        for (i = 0; i < M->GetNumberRows(); i++) {
            ptv[j] += ptM[i];
        }
    }

    for (i = 0; i < M->GetNumberColumns(); i++) {
        ptv[i] = ptv[i] / (float) M->GetNumberRows();
    }
}


void PLS::mean(Vector *v, Vector *retvector) {

    float *ptM, *ptv;
    int j;

    retvector->ResetVector();
    ptv = retvector->GetDataVector();
    ptM = v->GetDataVector();

    for (j = 0; j < v->GetSizeVector(); j++) {
        ptv[0] += ptM[j];
    }

    ptv[0] /= (float) v->GetSizeVector();

}


void PLS::std(Matrix *M, Vector *mean, Vector *retvector) {

    float *ptM, *ptret, *ptmean;
    int i, j;

    retvector->ResetVector();
    ptret = retvector->GetDataVector();
    ptmean = mean->GetDataVector();

    for (j = 0; j < M->GetNumberColumns(); j++) {
        ptM = M->CopyColumn(j);
        for (i = 0; i < M->GetNumberRows(); i++) {
            ptret[j] += (ptM[i] - ptmean[j]) * (ptM[i] - ptmean[j]);
        }
    }

    for (i = 0; i < M->GetNumberColumns(); i++) {
        ptret[i] = ptret[i] / (((float)M->GetNumberRows())- (float)1.0);
        ptret[i] = sqrt(ptret[i]);
        if (ptret[i] < EPSILON) {
            ptret[i] = 1;
        }
    }
}



void PLS::std(Vector *v, Vector *mean, Vector *retvector) {

    float *ptM, *ptret, *ptmean;
    int i;

    retvector->ResetVector();
    ptret = retvector->GetDataVector();
    ptmean = mean->GetDataVector();
    ptM = v->GetDataVector();

    for (i = 0; i < v->GetSizeVector(); i++) {
        ptret[0] += (ptM[i] - ptmean[0]) * (ptM[i] - ptmean[0]);
    }

    ptret[0] /= (((float)v->GetSizeVector())- (float)1.0);
    ptret[0] = sqrt(ptret[0]);

    if (ptret[0] < EPSILON) {
        ptret[0] = 1;
    }
}


void PLS::zscore(Matrix *M, Vector *mean, Vector *std) {

    float *ptM, *ptmean, *ptstd;
    int i,j;

    ptmean = mean->GetDataVector();
    ptstd = std->GetDataVector();

    for(j = 0; j < M->GetNumberColumns(); j++){
        ptM = M->CopyColumn(j);
        for (i = 0; i < M->GetNumberRows(); i++) {
            ptM[i] = (ptM[i] - ptmean[j]) / ptstd[j];
        }
    }
}


void PLS::ComputeWstar() {

    Matrix *tmp1, *tmp2, *tmp3;

    tmp1 = MatrixTransposed(P);
    tmp2 = MultiplyMatrices(tmp1, W);
    tmp3 = MatrixInverse(tmp2);
    Wstar = MultiplyMatrices(W, tmp3);

    delete tmp1;
    delete tmp2;
    delete tmp3;

}



void PLS::zscore(Vector *v, Vector *mean, Vector *std) {

    int i;
    float *ptM, *ptmean, *ptstd;

    ptmean = mean->GetDataVector();
    ptstd = std->GetDataVector();
    ptM = v->GetDataVector();

    for (i = 0; i < v->GetSizeVector(); i++) {
        ptM[i] = (ptM[i] - ptmean[0]) / ptstd[0];
    }

}


void PLS::runpls(Matrix *X, Vector *Y, int nfactor, char *OutputDir, float ExplainedX, float ExplainedY) {

    Matrix *U, *tmpM; /* Matrizes */

    Vector *t, *u, *t0, *Vcol, *Vrow, *w, *p, *C, *ymean, *ystd, *tmpV; /* Vetores */
    vector <int> selectedCols;

    double sumX2, sumY2, percX, percY, percAux, cumsum, cumsumY;
    float b_l, tmpscalar;  /* Escalares */
    float c, dt = 0;

    int i, j, kk;
    int maxsteps, step;
    int nsamples, nfeatures;

    // initially, clear current PLS model (if there is one)
    ClearPLS();

    nsamples = X->GetNumberRows();
    nfeatures = X->GetNumberColumns();

    Yorig = new Vector (nsamples);
    CopyVector(Y, Yorig);
    maxsteps = 100;

    Xmean = new Vector(nfeatures);
    Xstd = new Vector (nfeatures);

    mean(X, Xmean);
    std(X, Xmean, Xstd);
    zscore(X, Xmean, Xstd);

#if 0
    mean = umd_mean(X, rX, cX);
    printf("Medias finalizadas\n");
    std = umd_std(X, mean, rX, cX);
    printf("std finalizado\n");
    umd_zscore(X, mean, std, rX, cX);
    printf("z-scores finalizados\n");
#endif


    // Y
    ymean = new Vector (1);
    ystd = new Vector (1);
    mean(Y, ymean);
    std(Y, ymean, ystd);
    zscore(Y, ymean, ystd);


    C = new Vector (nfactor);
    T = new Matrix (nsamples, nfactor);
    U = new Matrix (nsamples, nfactor);
    P = new Matrix (nfeatures, nfactor);
    W = new Matrix (nfeatures, nfactor);
    b = new Vector (nfactor);


    t = new Vector (nsamples);
    u = new Vector (nsamples);
    t0 = new Vector (nsamples);
    Vcol = new Vector (nfeatures);
    Vrow = new Vector (nsamples);
    w = new Vector (nfeatures);
    p = new Vector (nfeatures);

    // compute square of the sum of X and Y
    sumY2 = 0;
    sumX2 = 0;

    for (i = 0; i < Y->GetSizeVector(); i++) {
        sumY2 += (double) (Y->GetElementVector(i) * Y->GetElementVector(i));
    }

    for (i = 0; i < X->GetNumberColumns(); i++) {
        for (j = 0; j < X->GetNumberRows(); j++) {
            sumX2 += (double) (X->GetElementMatrix(i, j) * X->GetElementMatrix(i, j));
        }
    }

    cumsum = 0;
    cumsumY = 0;

    for (i = 0; i < nfactor; i++) {

        normaliz(Y, t);
        CopyVector(t, u);
        step = 0;

           do {

            CopyVector(t, t0);

            MultiplyTransposeMatrixbyVector(X, u, Vcol);

            normaliz(Vcol, w);

            MultiplyMatrixbyVector(X, w, Vrow);

            normaliz(Vrow, t);

            tmpscalar = MultiplyVectorTransposedbyVector(Y, t);

            c = tmpscalar/tmpscalar;  //dummy step, because it normalizes a constant

            MultiplyVectorandScalar(Y, c, u);

            dt = 0;

            for (kk = 0; kk < nsamples; kk++) {
                dt += (t0->GetElementVector(kk) - t->GetElementVector(kk)) * (t0->GetElementVector(kk) - t->GetElementVector(kk));
            }

            step++;

        } while (dt > 0.000001 && step < maxsteps);

        MultiplyTransposeMatrixbyVector(X, t, p);
        b_l = MultiplyVectorTransposedbyVector(u, t);
        b->SetElementVector(i, b_l);

        for (j = 0; j < nfeatures; j++) {
            P->SetElementMatrix(i, j, p->GetElementVector(j));
            W->SetElementMatrix(i, j, w->GetElementVector(j));
        }

        for (j = 0; j < nsamples; j++) {
            T->SetElementMatrix(i, j, t->GetElementVector(j));
            U->SetElementMatrix(i, j, u->GetElementVector(j));
        }

        C->SetElementVector(i, c);

        SubtractFromMatrix(X, t, p);

        SubtractFromVector(Y, t, c, b->GetElementVector(i));

        percX = 0;
        percY = 0;
        percAux = 0;

        for (j = 0; j < T->GetNumberRows(); j++) {
            percAux += (double) (T->GetElementMatrix(i, j) * T->GetElementMatrix(i, j));
        }

        for (j = 0; j < P->GetNumberRows(); j++) {
            percY += (double) (P->GetElementMatrix(i, j) * P->GetElementMatrix(i, j));
        }

        percX = (percAux * percY) / sumX2;
        percY = (percAux * (double) (b->GetElementVector(i) * b->GetElementVector(i))) / sumY2;
        cumsum += percX;
        cumsumY += percY;

        if (cumsum >= ExplainedX) {
            cout << "reached percentage explained of X variable, stopping\n";
            nfactor = i + 1;
            break;
        }

        if (cumsum >= ExplainedY) {
            cout << "reached percentage explained of Y variable, stopping\n";
            nfactor = i + 1;
            break;
        }

     }

    // saving only number of factors actually used
    for (i = 0; i < nfactor; i++) {
        selectedCols.push_back(i);
    }


    tmpM = T->GetSelectedCols(&selectedCols); // implementar!!!!
    delete T;
    T = tmpM;

    tmpM = P->GetSelectedCols(&selectedCols);
    delete P;
    P = tmpM;

    tmpM = W->GetSelectedCols(&selectedCols);
    delete W;
    W = tmpM;

    tmpV = b->SelectElements(&selectedCols);
    delete b;
    b = tmpV;


    ComputeWstar();

    /* Cria um vetor auxiliar para zscore. */
    zdataV = new Vector (Wstar->GetNumberRows());

    /* Seta o numero maximo de fatores. */
    this->maxFactors = this->Wstar->GetNumberRows();

    /* Libera a memoria. */
    delete u;
    delete t0;
    delete Vcol;
    delete Vrow;
    delete w;
    delete p;
    delete C;
    delete U;
    delete ymean;
    delete ystd;
    delete t;
}


PLS::PLS() {

    Xmean = NULL;
    Xstd = NULL;
    T = NULL;
    P = NULL;
    W = NULL;
    Yorig = NULL;
    b = NULL;
    Wstar = NULL;
    zdataV = NULL;
    maxFactors = -1;
}


void PLS::SetMatrices(Matrix *W, Matrix *Wstar, Matrix *P, Vector *Xmean, Vector *Xstd, Vector *b) {

    ClearPLS();

    this->W = W;
    this->Wstar = Wstar;
    this->P = P;
    this->Xmean = Xmean;
    this->Xstd = Xstd;
    this->b = b;
    this->maxFactors = Wstar->GetNumberColumns();


    // create auxiliar vector for zscore
    zdataV = new Vector(Wstar->GetNumberRows());
}



PLS::~PLS() {

    ClearPLS();
}


void PLS::ClearPLS() {

    if (Xmean != NULL) {
        delete Xmean;
        Xmean = NULL;
    }

    if (Xstd != NULL) {
        delete Xstd;
        Xstd = NULL;
    }

    if (P != NULL) {
        delete P;
        P = NULL;
    }

    if (W != NULL) {
        delete W;
        W = NULL;
    }

    if (Yorig != NULL) {
        delete Yorig;
        Yorig = NULL;
    }

    if (T != NULL) {
        delete T;
        T = NULL;
    }

    if (b != NULL) {
        delete b;
        b = NULL;
    }

    if (Wstar != NULL) {
        delete Wstar;
        Wstar = NULL;
    }

    if (zdataV != NULL) {
        delete zdataV;
        zdataV = NULL;
    }

    maxFactors = -1;
}


void PLS::ClearExtraMatrices() {

    if (T != NULL) {
        delete T;
        T = NULL;
    }

    if (P != NULL) {
        delete P;
        P = NULL;
    }

    if (Yorig != NULL) {
        delete Yorig;
        Yorig = NULL;
    }

    if (W != NULL) {
        delete W;
        W = NULL;
    }

    if (b != NULL) {
        delete b;
        b = NULL;
    }
}




void PLS::ClearExtraMatricesPLSReg() {

    if (Yorig != NULL) {
        delete Yorig;
        Yorig = NULL;
    }

    if (b != NULL) {
        delete b;
        b = NULL;
    }

    if (T != NULL) {
        delete T;
        T = NULL;
    }

    if (P != NULL) {
        delete P;
        P = NULL;
    }

    if (W != NULL) {
        delete W;
        W = NULL;
    }

    if (Wstar != NULL) {
        delete Wstar;
        Wstar = NULL;
    }
}




void PLS::Projection(Vector *feat, Vector *retproj, int nfactors) {

    int i;

    ZscoreSSE(feat->GetDataVector(), Xmean->GetDataVector(), Xstd->GetDataVector(), zdataV->GetDataVector(), Wstar->GetNumberRows());

    for (i = 0; i < nfactors; i++) {
        retproj->v[i] = DotProductSSENotMultof4(zdataV->GetDataVector(), Wstar->CopyColumn(i), Wstar->GetNumberRows());
    }
}


void PLS::ExecuteZScore(float *feat, float *zscoreResult) {


    ZscoreSSE(feat, Xmean->GetDataVector(), Xstd->GetDataVector(), zscoreResult, Wstar->GetNumberRows());


}



void PLS::InitializePLSModel(Vector *Xmean, Vector *Xstd, Vector *Yorig, Vector *b, Matrix *T, Matrix *P,
                         Matrix *W, Matrix *Wstar) {

    ClearPLS();

    if (Xmean != NULL)
        this->Xmean = Xmean->Copy();

    if (Xstd != NULL)
        this->Xstd = Xstd->Copy();

    if (Yorig != NULL) {
        this->Yorig = Yorig->Copy();
    }

    if (b != NULL) {
        this->b = b->Copy();
    }

    if (T != NULL) {
        this->T = T->CopyMatrix();
    }

    if (P != NULL) {
        this->P = P->CopyMatrix();
    }

    if (W != NULL) {
        this->W = W->CopyMatrix();
    }

    if (Wstar != NULL) {

        this->Wstar = Wstar->CopyMatrix();

        /* Cria uma variavel auxiliar */
        zdataV = new Vector(Wstar->GetNumberRows());

        /*Seta o numero maximo de fatores */
        this->maxFactors = this->Wstar->GetNumberColumns();

    }
}


void PLS::CreatePLSModel(Matrix *X, Vector *Y, int nfactors) {

    Matrix *Xnew;
    Vector *Ynew;

    /* Copia as matrizes */
    Xnew = X->CopyMatrix();
    Ynew = Y->Copy();

    /* Roda o PLS */
    this->runpls(Xnew, Ynew, nfactors, NULL);

    /*Limpa os dados Utilizados */
    delete Xnew;
    delete Ynew;
}
