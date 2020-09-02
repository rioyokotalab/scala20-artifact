#include "utils.hpp"

const int64_t L3_DOUBLES = 3670016;
const int64_t L2_DOUBLES = 131072;

const int64_t B_SIZE = 8;
const int64_t VEC_SIZE = B_SIZE;
const int64_t extra = 5;

int64_t B_small = 1;
int64_t B_skinny = 200;
int64_t K_SIZE = 512;
Timer timer;
double packA_time, packB_time, mk_time;
double packA_V_time = 0, packB_S_time = 0, packB_U_time = 0, packA_S_time = 0;
int64_t batch_size, block_size, rank;
int64_t MB, NB, XB, YB, LDA_V, LDB_U, LDB_S, LDA_S;

double *malloc_aligned(int m, int n)
{
  double *ptr;

  ptr = (double*)aligned_alloc((size_t) GEMM_SIMD_ALIGN_SIZE, m*n*sizeof(double));
  if ( ptr == NULL ) {
    std::cout << "malloc_aligned failed. failure reason : ";
    if (errno == ENOMEM) {
      std::cout << "insufficient memory.";
    }
    else if (errno == EINVAL) {
      std::cout << "alignment not a power of two.";
    }

    exit( 1 );    
  }    
  return ptr;
}

double *malloc_aligned(int size) {
  return malloc_aligned(size , 1);
}

ticks getticks(void)
{
     unsigned a, d;
     asm("cpuid");
     asm volatile("rdtsc" : "=a" (a), "=d" (d));

     return (((ticks)a) | (((ticks)d) << 32));
}

void allocate_data(double ** &A, double ** &B, double ** &C,
                   double ** &Asim, double ** &Bsim, double ** &Csim,
                   int M, int N, int K) {
  int total_extra = extra * batch_size;
    
  Asim = (double**)malloc(sizeof(double*) * total_extra);
  Bsim = (double**)malloc(sizeof(double*) * total_extra);
  Csim = (double**)malloc(sizeof(double*) * total_extra);
  A = (double**)malloc(sizeof(double*) * batch_size);
  B = (double**)malloc(sizeof(double*) * batch_size);
  C = (double**)malloc(sizeof(double*) * batch_size);

  for (int b = 0; b < total_extra; ++b) {
    Asim[b] = (double*)malloc_aligned(M * K);
    Bsim[b] = (double*)malloc_aligned(K * N);
    Csim[b] = (double*)malloc_aligned(M * N);    
  }

  for (int b = 0; b < batch_size; ++b) {
    A[b] = Asim[rand() % total_extra];
    B[b] = Bsim[rand() % total_extra];
    C[b] = Csim[rand() % total_extra];
  }

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < M * K; ++i) { A[b][i] = 8.0; }
    for (int i = 0; i < K * N; ++i) { B[b][i] = 2.0; }
    for (int i = 0; i < M * N; ++i) { C[b][i] = 1.0; }
  }  
}

void allocate_data(double ** &A, double ** &Asim, int M, int N, double value) {
  int total_extra = extra * batch_size;
  std::random_device                  rand_dev;
  std::mt19937                        generator(rand_dev());

  Asim = (double**)aligned_alloc((size_t) GEMM_SIMD_ALIGN_SIZE, total_extra * sizeof(double*));
  A = (double**)aligned_alloc((size_t) GEMM_SIMD_ALIGN_SIZE, batch_size * sizeof(double*));

  for (int b = 0; b < total_extra; ++b) {
    Asim[b] = (double*)malloc_aligned(M, N);
  }

  for (int b = 0; b < batch_size; ++b) {
    std::uniform_int_distribution<int>  distr(b * extra,
                                             b * extra + extra-1);
    A[b] = Asim[distr(generator)];
  }

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < M * N; ++i) {
      A[b][i] = value == 0 ? 0 : (double(rand()) / RAND_MAX);
    }
  }    
}

void free_data(double ** A, double ** Asim) {
  int total_extra = extra * batch_size;

  for (int b = 0; b < total_extra; ++b) {
    free(Asim[b]);
  }
  free(Asim);
  free(A);
}

void free_data(double ** A, double ** B, double ** C, double ** Asim, double ** Bsim, double ** Csim) {
  int total_extra = extra * batch_size;

  for (int b = 0; b < total_extra; ++b) {
    free(Asim[b]);
    free(Bsim[b]);
    free(Csim[b]);
  }
  free(Asim); free(Bsim); free(Csim);
}

void print (double * mat, int M , int N, int LDmat) {
  for (int i = 0; i < M; ++i) {

    for (int j = 0; j < N; ++j) {
      std::cout << std::setw(7) << std::setprecision(4) << mat[i * LDmat + j] << " ";
      if (j % VEC_SIZE == VEC_SIZE-1)
        std::cout << std::setw(7) << " | ";
    }
    if (i % VEC_SIZE == VEC_SIZE - 1) {
      std::cout << "\n------------------------------------------------------------\n";
    }
    std::cout << std::endl;
    
  }

}

