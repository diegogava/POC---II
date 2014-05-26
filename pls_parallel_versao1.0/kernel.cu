#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define numThreadLinha 16
#include "cuda_runtime.h"
#include "matrix.h"

__global__ void mulCUDA(float *m1, float *m2, float* result, int numTotEleLinha);

/* Multiplica duas matrizes. */
Matrix *MultiplyMatrices(Matrix *m1, Matrix *m2){

    Matrix *result;
	unsigned int numTotEleLinha, numBlocos, tamanho_bytes, nThreads;
	float *ga, *gb, *gc;


    if(m1->GetNumberColumns() != m2->GetNumberRows()){
        cout << "Nao eh possivel multiplicar tais matrizes!\n";
        exit(2);
    }

    result = new Matrix(m1->GetNumberRows(),m2->GetNumberColumns());

	numTotEleLinha = m1->GetNumberRows();
	 
	/* define numero de threads por numThreadLinha */
  dim3 nThreadsPorBloco (numThreadLinha,numThreadLinha);
 
  /* define numero de blocos */
 
  numBlocos = numTotEleLinha / numThreadLinha;
  dim3 nBlocosPorGrid(numBlocos,numBlocos);
  nThreads = numTotEleLinha*numTotEleLinha;
 
  /* alocando memoria global da GPU */
  tamanho_bytes = nThreads * sizeof(float);
  cudaMalloc((void **) &ga, tamanho_bytes);
  cudaMalloc((void **) &gb, tamanho_bytes);
  cudaMalloc((void **) &gc, tamanho_bytes);
 
  /* Copia dados da RAM para a memoria global da GPU */
  cudaMemcpy(ga, m1->m, tamanho_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gb, m2->m, tamanho_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gc, result->m, tamanho_bytes, cudaMemcpyHostToDevice);
 
 
  mulCUDA<<<nBlocosPorGrid,nThreadsPorBloco>>>(ga, gb, gc, m1->GetNumberRows());
 
 
 
  /* Copia o resultado da memoria global da GPU ah RAM */
  cudaMemcpy(result->m, gc, tamanho_bytes, cudaMemcpyDeviceToHost);
 
  
 
  cudaFree(ga);
  cudaFree(gb);
  cudaFree(gc);

    return result;

}


/*   Kernel CUDA */
__global__ void mulCUDA(float *m1, float *m2, float * result, int numlinhas){

  int i, j, k;

  i = blockIdx.y * numThreadLinha + threadIdx.y;
  j = blockIdx.x * numThreadLinha + threadIdx.x;
  
  float soma = 0.0;
  for (k = 0; k < numlinhas; ++k){
    soma += m1[i*numlinhas + k] * m2[k*numlinhas + j];
  }
  result[i*numlinhas + j] = soma;
}