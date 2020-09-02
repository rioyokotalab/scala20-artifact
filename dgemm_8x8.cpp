#include "utils.hpp"

#define UPDATE_BLOCK(R1, R2, REGISTER)            \
                                                \
  VMOVUPD(MEM(REGISTER), ZMM(R1))               \
  VMOVUPD(MEM(REGISTER, 64), ZMM(R2))           \
  LEA(REGISTER, MEM(REGISTER, 2, 64))

#define KI_SUBITER(n)                           \
                                                \
  VMOVAPD(ZMM(15), MEM(RBX, (n * 8) * 8))       \
                                                \
  VBROADCASTSD(ZMM(23), MEM(RAX, (8 * n + 0) * 8))  \
  VBROADCASTSD(ZMM(22), MEM(RAX, (8 * n + 1) * 8))  \
  VFMADD231PD(ZMM(31), ZMM(23), ZMM(15))        \
  VFMADD231PD(ZMM(30), ZMM(22), ZMM(15))        \
                                                \
  VBROADCASTSD(ZMM(21), MEM(RAX, (8 * n + 2) * 8))  \
  VBROADCASTSD(ZMM(20), MEM(RAX, (8 * n + 3) * 8))  \
  VFMADD231PD(ZMM(29), ZMM(21), ZMM(15))        \
  VFMADD231PD(ZMM(28), ZMM(20), ZMM(15))        \
                                                \
  VBROADCASTSD(ZMM(19), MEM(RAX, (8 * n + 4) * 8))  \
  VBROADCASTSD(ZMM(18), MEM(RAX, (8 * n + 5) * 8))  \
  VFMADD231PD(ZMM(27), ZMM(19), ZMM(15))        \
  VFMADD231PD(ZMM(26), ZMM(18), ZMM(15))        \
                                                \
  VBROADCASTSD(ZMM(17), MEM(RAX, (8 * n + 6) * 8))  \
  VBROADCASTSD(ZMM(16), MEM(RAX, (8 * n + 7) * 8))  \
  VFMADD231PD(ZMM(25), ZMM(17), ZMM(15))        \
  VFMADD231PD(ZMM(24), ZMM(16), ZMM(15))        \

#define BROADCAST_A_S_ROW(n)                    \
  VBROADCASTSD(ZMM(23), MEM(RAX, (0 * 8)) )     \
  VBROADCASTSD(ZMM(14), MEM(RAX, (1 * 8)) )     \
  VBROADCASTSD(ZMM(13), MEM(RAX, (2 * 8)) )     \
  VBROADCASTSD(ZMM(12), MEM(RAX, (3 * 8)) )     \
  VBROADCASTSD(ZMM(11), MEM(RAX, (4 * 8)) )     \
  VBROADCASTSD(ZMM(10), MEM(RAX, (5 * 8)) )     \
  VBROADCASTSD(ZMM(9),  MEM(RAX, (6 * 8)) )     \
  VBROADCASTSD(ZMM(8),  MEM(RAX, (7 * 8)) )     \
  LEA(RAX, MEM(RAX, 64))

#define FMA_EXN_ROW(R1)                           \
  VFMADD231PD(ZMM(15), ZMM(23), ZMM(R1))        \
  VFMADD231PD(ZMM(16), ZMM(14), ZMM(R1))        \
  VFMADD231PD(ZMM(17), ZMM(13), ZMM(R1))        \
  VFMADD231PD(ZMM(18), ZMM(12), ZMM(R1))        \
  VFMADD231PD(ZMM(19), ZMM(11), ZMM(R1))        \
  VFMADD231PD(ZMM(20), ZMM(10), ZMM(R1))        \
  VFMADD231PD(ZMM(21), ZMM(9),  ZMM(R1))         \
  VFMADD231PD(ZMM(22), ZMM(8),  ZMM(R1))

#define UPDATE_EXN(R1)                                          \
  VBROADCASTSD(ZMM(23), MEM(RAX, (0 * 8)))                      \
  VBROADCASTSD(ZMM(14), MEM(RAX, (1 * 8)))                      \
  VFMADD231PD(ZMM(15), ZMM(23), ZMM(R1))                        \
  VFMADD231PD(ZMM(16), ZMM(14), ZMM(R1))                        \
                                                                \
  VBROADCASTSD(ZMM(13), MEM(RAX, (2 * 8)))                      \
  VBROADCASTSD(ZMM(12), MEM(RAX, (3 * 8)))                      \
  VFMADD231PD(ZMM(17), ZMM(13), ZMM(R1))                        \
  VFMADD231PD(ZMM(18), ZMM(12), ZMM(R1))                        \
                                                                \
  VBROADCASTSD(ZMM(11), MEM(RAX, (4 * 8)))                      \
  VBROADCASTSD(ZMM(10), MEM(RAX, (5 * 8)))                      \
  VFMADD231PD(ZMM(19), ZMM(11), ZMM(R1))                        \
  VFMADD231PD(ZMM(20), ZMM(10), ZMM(R1))                \  
\
VBROADCASTSD(ZMM(9),  MEM(RAX, (6 * 8)))       \
VBROADCASTSD(ZMM(8),  MEM(RAX, (7 * 8)))       \
VFMADD231PD(ZMM(21), ZMM(9),  ZMM(R1))         \
VFMADD231PD(ZMM(22), ZMM(8),  ZMM(R1))         \
\
LEA(RAX, MEM(RAX, 64))


void dgemm_8x8(double *GXYb, int mc, int nc, int xc, int yc, double * packA_V,
               double * packB_S, double *packB_U, double *packA_S) {
  timer.start();
  // micro kernel
  int64_t k_iter = K_SIZE / 4;
  double * EXN = (double*)malloc_aligned(VEC_SIZE, VEC_SIZE);

  BEGIN_ASM()

    VXORPD(YMM(31), YMM(31), YMM(31))
    VXORPD(YMM(30), YMM(30), YMM(30))
    VXORPD(YMM(29), YMM(29), YMM(29))
    VXORPD(YMM(28), YMM(28), YMM(28))
    VXORPD(YMM(27), YMM(27), YMM(27))
    VXORPD(YMM(26), YMM(26), YMM(26))
    VXORPD(YMM(25), YMM(25), YMM(25))    
    VXORPD(YMM(24), YMM(24), YMM(24))

    MOV(RSI, VAR(k_iter))
    MOV(RAX, VAR(packA_V))
    MOV(RBX, VAR(packB_U))

    // Update CMN two rows/cols at a time.
    LABEL(BEGIN_LOOP_KI)

      KI_SUBITER(0)
      KI_SUBITER(1)
      KI_SUBITER(2)
      KI_SUBITER(3)
    
      LEA(RAX, MEM(RAX, 4 * 8 * 8))
      LEA(RBX, MEM(RBX, 4 * 8 * 8))
    DEC(RSI)
    JNZ(BEGIN_LOOP_KI)

    // UPDATE EXN
    VXORPD(YMM(15), YMM(15), YMM(15))
    VXORPD(YMM(16), YMM(16), YMM(16))
    VXORPD(YMM(17), YMM(17), YMM(17))
    VXORPD(YMM(18), YMM(18), YMM(18))
    VXORPD(YMM(19), YMM(19), YMM(19))  MOV(RAX, VAR(packA_S))
    VXORPD(YMM(20), YMM(20), YMM(20))
    VXORPD(YMM(21), YMM(21), YMM(21))
    VXORPD(YMM(22), YMM(22), YMM(22))

    UPDATE_EXN(31)
    UPDATE_EXN(30)
    UPDATE_EXN(29)
    UPDATE_EXN(28)
    UPDATE_EXN(27)
    UPDATE_EXN(26)
    UPDATE_EXN(25)
    UPDATE_EXN(24)
    
    MOV(RCX, VAR(EXN))

    UPDATE_BLOCK(15, 16, RCX)
    UPDATE_BLOCK(17, 18, RCX)
    UPDATE_BLOCK(19, 20, RCX)
    UPDATE_BLOCK(21, 22, RCX)    
    
  END_ASM(
          :                   // output operands
          :                   // input operands
          [packA_V] "m" (packA_V),
          [packB_S] "m" (packB_S),
          [packB_U] "m" (packB_U),
          [packA_S] "m" (packA_S),
          [K_SIZE] "m" (K_SIZE),
          [k_iter] "m" (k_iter),
          [EXN] "m" (EXN)
          :                  // registers
          "rax", "rbx", "rcx", "rdx", "rsi", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory");

  for (int xi = 0; xi < VEC_SIZE; ++xi) {
    for (int yi = 0; yi < VEC_SIZE; ++yi) {
      for (int ni = 0; ni < VEC_SIZE; ++ni) {
        GXYb[(xc + xi) * LDA_S + yc + yi] += EXN[xi * VEC_SIZE + ni] * packB_S[nc + ni * VEC_SIZE + yc + yi];
      }
    }
  } // end of micro kernel

  timer.stop();
  mk_time += timer.time_ns();
}
