#include <iostream>
#include "manual_mult.hpp"
#include <cstring>
#include <omp.h>

int av_size;

inline void packing_B_S(double *packB_S, double **B_S, int av_c, int av_pack_size) {
  int batch_offset = av_c * B_skinny;
  
  timer.start();
  #pragma omp parallel for
  for (int batch = 0; batch < av_pack_size; ++batch) {
    double * tempBS, * temp;    
    // pack B_S
    temp = B_S[batch + batch_offset];
    tempBS = packB_S + batch * VEC_SIZE * VEC_SIZE;

    for (int nbi = 0; nbi < rank; ++nbi) {
      for (int i = 0; i < VEC_SIZE; ++i) {
        *tempBS++ = *temp++;
      }
    }
  }
  timer.stop();
  packB_S_time += timer.time_ns();
}

inline void packing_B_U(double * packB_U, double ** B_U, int batch_offset, int bu_pack_size) {
  timer.start();
  double *tempBU = packB_U;
  double * temp;

  for (int batch = 0; batch < bu_pack_size; ++batch) {
    temp = B_U[batch + batch_offset];
    for (int k = 0; k < block_size; ++k) {
      *tempBU = *temp;
      *(tempBU + 1) = *(temp + 1);
      *(tempBU + 2) = *(temp + 2);
      *(tempBU + 3) = *(temp + 3);
      *(tempBU + 4) = *(temp + 4);
      *(tempBU + 5) = *(temp + 5);
      *(tempBU + 6) = *(temp + 6);
      *(tempBU + 7) = *(temp + 7);
                
      tempBU += 8;
      temp += 8;
    }
  }

  timer.stop();
  packB_U_time += timer.time_ns();
}

inline void packing_A_V(double * packA_V, double ** A_V, int batch_offset, int bu_pack_size) {
  timer.start();
  double * tempAV, * temp;
  tempAV = packA_V;
  double * temp_init = packA_V;
  
  for (int batch = 0; batch < bu_pack_size; ++batch) {
    temp = A_V[batch + batch_offset];

    for (int i =  0; i < VEC_SIZE; ++i) {
      tempAV = temp_init + i;
      for (int k = 0; k < block_size; k++) {              
        *tempAV = *temp++;
        tempAV += VEC_SIZE;
      }
    }
    temp_init += av_size;
  }
  timer.stop();
  packA_V_time += timer.time_ns();
}

inline void packing_A_S(double *packA_S, double **A_S, int av_c, int av_pack_size) {
  int batch_offset = av_c * B_skinny;
  
  timer.start();
  #pragma omp parallel for
  for (int batch = 0; batch < av_pack_size; ++batch) {
    double * tempAS, * temp, * temp_init;    
    // pack B_S
    temp_init = packA_S + batch * VEC_SIZE * VEC_SIZE;
    temp = A_S[batch + batch_offset];
    tempAS = temp_init;

    for (int i = 0; i < VEC_SIZE; ++i) {
      tempAS = temp_init + i;
      for (int mbi = 0; mbi < rank; ++mbi) {
        *tempAS = *temp++;
        tempAS += VEC_SIZE;
      }
    }

  }
  timer.stop();
  packA_S_time += timer.time_ns();
}


void macro_kernel(double *GXYb, double * packA_V, double * packB_S,
                  double * packB_U, double * packA_S) {
  for (int mc = 0; mc < MB; mc += VEC_SIZE) {
    for (int nc = 0; nc < NB; nc += VEC_SIZE) {
      for (int xc = 0; xc < rank; xc += VEC_SIZE) {
        for (int yc = 0; yc < rank; yc += VEC_SIZE) {
            
          // micro kernel
          dgemm_8x8(GXYb, mc, nc, xc, yc, packA_V, packB_S, packB_U, packA_S);
        }
      }
    }
  }
}

inline void determine_batch_sizes() {
  int av_batch;
  int bu_batch;

  if (B_small == -1) {
    bu_batch = floor(65536.0 / (block_size * rank));
  }
  else {
    bu_batch = B_small;
  }

  if (B_skinny == -1) {
    av_batch = floor(1835008.0 / (rank * rank));
    if (batch_size < av_batch) {
      av_batch = batch_size;
    }
  }
  else {
    av_batch = B_skinny;
  }

  int diff = (av_batch % bu_batch);

  // Make sure that A_V and B_U batch sizes are perfectly divisible.
  B_skinny = av_batch - diff;
  B_small = bu_batch;
}

void manual_mult(double ** A_V, double ** A_S, double ** B_S, double ** B_U, double ** GXY) {
  double *GXYb, *packB_S, *packA_S;
  double *first_packB_S, *first_packA_S;

  determine_batch_sizes();

  LDA_V = block_size, LDB_U = rank, LDB_S = rank, LDA_S = rank;
  K_SIZE = block_size;
  av_size = block_size * VEC_SIZE;

  if (rank > VEC_SIZE) {
    MB = VEC_SIZE * 2;
    NB = VEC_SIZE * 2;
    XB = VEC_SIZE * 2;
    YB = VEC_SIZE * 2;
  }
  else {
    MB = VEC_SIZE;
    NB = VEC_SIZE;
    XB = VEC_SIZE;
    YB = VEC_SIZE;
  }

  packB_S = (double*)malloc_aligned(rank * B_skinny, VEC_SIZE);
  packA_S = (double*)malloc_aligned(VEC_SIZE, rank * B_skinny);

  first_packB_S = packB_S;
  first_packA_S = packA_S;

  const int av_b = (batch_size + B_skinny - 1) / B_skinny;
  const int _av_c = batch_size % B_skinny;
  int av_pack_size;

  // outer looping
  for (int av_c = 0; av_c < av_b; av_c++) {
    av_pack_size = (av_c != av_b - 1 || _av_c == 0) ? B_skinny : _av_c;
    
    packB_S = first_packB_S;
    packA_S = first_packA_S;

    packing_B_S(packB_S, B_S, av_c, av_pack_size);
    packing_A_S(packA_S, A_S, av_c, av_pack_size);

    const int bu_b = (av_pack_size + B_small - 1) / B_small;
    const int _bu_c = av_pack_size % B_small;

    #pragma omp parallel for
    for (int bu_c = 0; bu_c < bu_b; bu_c++) {
      double *packB_U, *packA_V;
      double *first_packB_U, *first_packA_V;
      double *local_packB_S, *local_packA_S;
      int batch_offset;
      int bu_pack_size;

      bu_pack_size = (bu_c != bu_b - 1 || _bu_c == 0) ? B_small : _bu_c;
      
      packB_U = (double*)malloc_aligned(bu_pack_size * block_size, VEC_SIZE);
      packA_V = (double*)malloc_aligned(VEC_SIZE, bu_pack_size * block_size);

      first_packB_U = packB_U;
      first_packA_V = packA_V;
      
      local_packB_S = first_packB_S + bu_c * B_small * VEC_SIZE * VEC_SIZE;
      local_packA_S = first_packA_S + bu_c * B_small * VEC_SIZE * VEC_SIZE;
      batch_offset = av_c * B_skinny + bu_c * B_small;

      packing_A_V(packA_V, A_V, batch_offset, bu_pack_size);
      packing_B_U(packB_U, B_U, batch_offset, bu_pack_size);
      
      for (int nbatch = batch_offset; nbatch < batch_offset + bu_pack_size; ++nbatch) {
        GXYb = GXY[nbatch];
        macro_kernel(GXYb, packA_V, local_packB_S, packB_U, local_packA_S);
        
        packA_V += VEC_SIZE * block_size;
        local_packB_S += VEC_SIZE * VEC_SIZE;
        local_packA_S += VEC_SIZE * VEC_SIZE;
        packB_U += VEC_SIZE * block_size;
      }

      free(first_packB_U);
      free(first_packA_V);
    }
  }
}
