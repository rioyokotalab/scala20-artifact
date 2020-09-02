#include "utils.hpp"
#include "mkl.h"
#include <omp.h>

void mkl_mult(double ** A_V, double ** A_S, double ** B_S, double ** B_U, double ** GXY);
void mkl_batched_mult(double ** A_V, double ** A_S, double ** B_S, double ** B_U,
                      double ** GXY, double ** CMN, double ** EXN);
