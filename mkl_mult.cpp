#include "mkl_mult.hpp"

void mkl_mult(double ** A_V, double ** A_S, double ** B_S, double ** B_U, double ** GXY) {
  double ** CMN = (double**)malloc(sizeof(double*) * batch_size);
  double ** EXN = (double**)malloc(sizeof(double*) * batch_size);
  for (int b = 0; b < batch_size; ++b) {
    CMN[b] = (double*)calloc(rank * rank, sizeof(double));
    EXN[b] = (double*)calloc(rank * rank, sizeof(double));
  }

  double *packAS = (double*)malloc(sizeof(double) * rank * rank * batch_size);
  double *packBS = (double*)malloc(sizeof(double) * rank * rank * batch_size);
  double *temp_packAS, *temp_packBS;
  
  // Pack small matrices into a long array for fair comparison.
  #pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    int offset = b * rank * rank;
    temp_packAS = packAS + offset;
    temp_packBS = packBS + offset;
      
    for (int i = 0; i < rank * rank; ++i) {
      temp_packAS[i] = A_S[b][i];
      temp_packBS[i] = B_S[b][i];
    }
  }


  #pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    double * A_Vb = A_V[b];
    double * A_Sb = &packAS[b * rank * rank];
    double * B_Sb = &packBS[b * rank * rank];
    double * B_Ub = B_U[b];
    double * GXYb = GXY[b];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, rank, block_size, 1.0, A_Vb, block_size, B_Ub, rank, 0.0, CMN[b], rank);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, rank, rank, 1.0, A_Sb, rank, CMN[b], rank, 0.0, EXN[b], rank);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, rank, rank, 1.0, EXN[b], rank, B_Sb, rank, 0.0, GXYb, rank);
  }
  
  for (int b = 0; b < batch_size; ++b) {
    free(CMN[b]); free(EXN[b]);
  }
  free(CMN); free(EXN);
  free(packAS);
  free(packBS);

}

void mkl_batched_mult(double ** A_V, double ** A_S, double ** B_S, double ** B_U, double ** GXY,
                      double **CMN, double **EXN) {

  int group_count = 1;
  int group_size[1] = { int(batch_size) };
  const double *a_array[batch_size], *b_array[batch_size];
  double *c_array[batch_size];
  int lda_array[batch_size], ldb_array[batch_size], ldc_array[batch_size];
  CBLAS_TRANSPOSE transa_array[batch_size], transb_array[batch_size];
  int m_array[batch_size], n_array[batch_size], k_array[batch_size];
  double alpha_array[batch_size], beta_array[batch_size];

  for (int b = 0; b < batch_size; ++b) {
    a_array[b] = A_V[b];
    b_array[b] = B_U[b];
    c_array[b] = CMN[b];
    lda_array[b] = block_size;
    ldb_array[b] = rank;
    ldc_array[b] = rank;
    transa_array[b] = CblasNoTrans;
    transb_array[b] = CblasNoTrans;
    m_array[b] = rank;
    n_array[b] = rank;
    k_array[b] = block_size;
    alpha_array[b] = 1.0;
    beta_array[b] = 0.0;
  }

  cblas_dgemm_batch(CblasRowMajor, transa_array, transb_array, m_array, n_array, k_array,
                    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array,
                    ldc_array, group_count, group_size);
    

  // BLOCK product. EXN += C * S2
  for (int b = 0; b < batch_size; ++b) {
    a_array[b] = A_S[b];
    b_array[b] = CMN[b];
    c_array[b] = EXN[b];
    lda_array[b] = rank;
    ldb_array[b] = rank;
    ldc_array[b] = rank;
    m_array[b] = rank;
    n_array[b] = rank;
    k_array[b] = rank;
  }

  cblas_dgemm_batch(CblasRowMajor, transa_array, transb_array, m_array, n_array, k_array,
                    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array,
                    ldc_array, group_count, group_size);

  
  for (int b = 0; b < batch_size; ++b) {
    a_array[b] = EXN[b];
    b_array[b] = B_S[b];
    c_array[b] = GXY[b];
  }
  
  cblas_dgemm_batch(CblasRowMajor, transa_array, transb_array, m_array, n_array, k_array,
                    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array,
                    ldc_array, group_count, group_size);


}
