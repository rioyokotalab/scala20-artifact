inline void dgemm_8x8(double *GXYb, int mc, int nc, int xc, int yc, double * packA_V,
                      double * packB_S, double *packB_U, double *packA_S);

void dumb_kernel(double *GXYb, int mc, int nc, int xc, int yc, double * packA_V,
               double * packB_S, double *packB_U, double *packA_S);

