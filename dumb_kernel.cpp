#include "utils.hpp"


void dumb_kernel(double *GXYb, int mc, int nc, int xc, int yc, int mb, int nb, double * packA_V,
                 double * packB_S, double *packB_U, double *packA_S) {
  double cmn, exn;
  for (int mi = 0; mi < B_SIZE; ++mi) {
    for (int ni = 0; ni < B_SIZE; ++ni) {
      for (int ki = 0; ki < K_SIZE; ++ki) {
        for (int xi = 0; xi < B_SIZE; ++xi) {
          for (int yi = 0; yi < B_SIZE; ++yi) {
            cmn = packA_V[(mc * block_size + mi) + ki * VEC_SIZE] *
              packB_U[ki * VEC_SIZE + nb + nc + ni];
            exn = packA_S[(mi + mc + mb) * VEC_SIZE + (xc + xi) ] * cmn;
            GXYb[(xc + xi) * LDA_S + yc + yi] += exn * packB_S[nc + ni * VEC_SIZE + (yc + yi)];
          }
        }
      }
    }
  }
}
