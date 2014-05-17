/*
   http://www.liv.ic.unicamp.br/~wschwartz/softwares.html

   Copyright (C) 2010-2011 William R. Schwartz

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   William R. Schwartz williamrobschwartz [at] gmail.com
*/

#include "vector.h"
#include "matrix.h"
#include "pls_sequential.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {

    Matrix *X;
    Vector *Y;
    PLS *P;



    X = new Matrix ("Input_data/file_matrix_50");
    Y = new Vector ("Input_data/file_vector_50");

    P = new PLS();

    P->runpls(X,Y,50,NULL);

    delete X;
    delete Y;
    delete P;

    return 0;

}
