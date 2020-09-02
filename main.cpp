#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "utils.hpp"
#include "mkl_mult.hpp"
#include "manual_mult.hpp"

double flops;

void test_everything(double **A_V, double **B_U, double **A_S, double **B_S, double **GXY_mkl,
                     double **GXY_manual, double ** GXY_batched, std::ofstream& output) {
  Timer t;
  double mkl_time, manual_time, batched_time;
  t.start();    
  mkl_mult(A_V, A_S, B_S, B_U, GXY_mkl);
  t.stop();
  mkl_time = t.time_ms();

  t.start();
  manual_mult(A_V, A_S, B_S, B_U, GXY_manual);
  t.stop();
  manual_time = t.time_ms();

  double ** CMN = (double**)malloc(sizeof(double*) * batch_size);
  double ** EXN = (double**)malloc(sizeof(double*) * batch_size);
  for (int b = 0; b < batch_size; ++b) {
    CMN[b] = (double*)calloc(rank * rank, sizeof(double));
    EXN[b] = (double*)calloc(rank * rank, sizeof(double));
  }

  t.start();
  mkl_batched_mult(A_V, A_S, B_S, B_U, GXY_batched, CMN, EXN);
  t.stop();
  batched_time = t.time_ms();
  for (int b = 0; b < batch_size; ++b) {
    free(CMN[b]); free(EXN[b]);
  }
  free(CMN); free(EXN);

  double mkl_norm, manual_norm, batched_norm;
  double batch_norm_diff = 0;

  for (int b = 0; b < batch_size; ++b) {
    mkl_norm = 0;
    manual_norm = 0;
    batched_norm = 0;
    for (int i = 0; i < rank; ++i) {
      for (int j = 0; j < rank; ++j) {
        mkl_norm += pow(GXY_mkl[b][i * rank + j], 2);
        manual_norm += pow(GXY_manual[b][i * rank + j], 2);
        batched_norm += pow(GXY_batched[b][i * rank + j], 2);
      }
    }

    mkl_norm = sqrt(mkl_norm);
    manual_norm = sqrt(manual_norm);
    batched_norm = sqrt(batched_norm);
    batch_norm_diff += (manual_norm - mkl_norm);
  }

  double mkl_gflops = flops / mkl_time / 1e6;
  double manual_gflops = flops / manual_time / 1e6;
  double mkl_batched_gflops = flops / batched_time / 1e6;

  int tid = 80;
  if (std::getenv("OMP_NUM_THREADS")) {
    tid = atoi(std::getenv("OMP_NUM_THREADS"));
  }
  tid = 1;
  
  mk_time = mk_time / 1e6;  
  packA_V_time = packA_V_time / tid / 1e6;
  packB_S_time = packB_S_time / 1e6;
  packB_U_time = packB_U_time / tid / 1e6;
  packA_S_time = packA_S_time / 1e6;

  std::cout << "--------------------------\n";
  std::cout << "Batch size:      " << batch_size << std::setw(3) << std::endl;
  std::cout << "Rank:            " << rank << std::endl;
  std::cout << "Block:           " << block_size << std::endl;
  std::cout << "B_skinny:       " << B_skinny
            << "\nB_small:       " << B_small << std::endl;
  std::cout << "MKL time:        " << mkl_time << std::setw(3)
            << " Gflops: " << mkl_gflops << std::endl;
  std::cout << "MKL(B) time:     " << batched_time << " Gflops: " << mkl_batched_gflops <<  std::endl;
  std::cout << "Manual time:     " << manual_time << std::setw(3) << " Gflops: " << manual_gflops << std::endl;
  std::cout << " mk_time: " << mk_time
            << "\n packA_V time: " << packA_V_time
            << "\n packB_S time: " << packB_S_time
            << "\n packB_U time: " << packB_U_time
            << "\n packA_S time: " << packA_S_time
            <<  std::endl;
  std::cout << "Batch norm diff: " << batch_norm_diff << std::setw(3) <<std::endl;
  std::cout << "--------------------------\n";


  if (output.is_open()) {
    output << batch_size << "," << block_size << "," << rank << "," << mkl_time
           << "," << mkl_gflops << "," << manual_time << "," << manual_gflops
           << "," << mk_time << "," << packA_V_time << "," << packB_S_time
           << "," << packB_U_time << "," << packA_S_time
           << "," << B_skinny << "," << B_small << std::endl;

    output.close();
  }
}

int main(int argc, char *argv[])
{
  double **A_V, **B_U, **A_S, **B_S, **GXY_mkl, **GXY_manual, **GXY_batched,
    **A_V_sim, **B_U_sim, **A_S_sim, **B_S_sim, **GXY_mkl_sim, **GXY_manual_sim,
    **GXY_batched_sim;
  Timer t;
  double time;
  std::ofstream output;
  int choice;
  packA_time = packB_time = mk_time = 0;

  B_skinny = -1;
  B_small = -1;
  
  if (argc > 1) {
    batch_size = atoi(argv[1]);
    block_size = atoi(argv[2]);
    rank = atoi(argv[3]);
    output.open(argv[4], std::ios::app | std::ios::out);
    B_small = atoi(argv[5]);
    choice = argc > 6 ? atoi(argv[6]) : -1;
    if (argc > 7) {
      B_skinny = atoi(argv[7]);
    }    
  }
  else {
    return 1;
  }

  if (batch_size < B_skinny || batch_size < B_small) {
    std::cout << "small batch!\n";
    abort();
  }

  int tnum = 80;
  int mklt = 1;
  if (std::getenv("OMP_NUM_THREADS")) {
    tnum = atoi(std::getenv("OMP_NUM_THREADS"));
  }
  if (std::getenv("MKL_NUM_THREADS")) {
    mklt = atoi(std::getenv("MKL_NUM_THREADS"));
  }

  if (mklt > tnum) {
    tnum = mklt;
  }

  flops = batch_size * (2 * rank * rank * block_size + 2 * rank * rank * rank + 2 * rank * rank * rank);

  srand(1000);

  allocate_data(A_V, A_V_sim, rank, block_size);
  allocate_data(A_S, A_S_sim, rank, rank);
  allocate_data(B_S, B_S_sim, rank, rank);
  allocate_data(B_U, B_U_sim, block_size, rank);
  allocate_data(GXY_mkl, GXY_mkl_sim, rank, rank, 0);
  allocate_data(GXY_manual, GXY_manual_sim, rank, rank, 0);
  allocate_data(GXY_batched, GXY_batched_sim, rank, rank, 0);

  double gflops;

  switch (choice) {
  case 0: {
    t.start();    
    mkl_mult(A_V, A_S, B_S, B_U, GXY_mkl);
    t.stop();
    time = t.time_ms();
    break;
  }
  case 1: {
    double **CMN, **EXN, **CMN_sim, **EXN_sim;
    allocate_data(CMN, CMN_sim, rank, rank);
    allocate_data(EXN, EXN_sim, rank, rank);
    
    t.start();
    mkl_batched_mult(A_V, A_S, B_S, B_U, GXY_batched, CMN, EXN);
    t.stop();
    time = t.time_ns() / 1e9;

    free_data(EXN, EXN_sim);
    free_data(CMN, CMN_sim);
    break;
  }
  case 2: {
    t.start();
    manual_mult(A_V, A_S, B_S, B_U, GXY_manual);
    t.stop();
    time = t.time_ms();

    mk_time = mk_time / 1e6;
    packA_V_time = packA_V_time / 1e6;
    packB_S_time = packB_S_time / 1e6;
    packB_U_time = packB_U_time / 1e6;
    packA_S_time = packA_S_time / 1e6;

    gflops = flops / time / 1e6;

    break;
  }
  default:
    test_everything(A_V, B_U, A_S, B_S, GXY_mkl, GXY_manual, GXY_batched, output);
    return 0;
  }
  
  if (output.is_open()) {
    std::cout << "--------------------------\n";
    std::cout << "Batch size:      " << batch_size << std::setw(3) << std::endl;
    std::cout << "Rank:            " << rank << std::endl;
    std::cout << "Block:           " << block_size << std::endl;
    std::cout << "B_skinny:        " << B_skinny
            << "\nB_small  :       " << B_small << std::endl;
    std::cout << "Total Time:      " << time << std::setw(3)
              << " Gflops: " << gflops << std::endl;
    std::cout << " mk_time: " << mk_time
              << "\n packA_V time: " << packA_V_time
              << "\n packB_S time: " << packB_S_time
              << "\n packB_U time: " << packB_U_time
              << "\n packA_S time: " << packA_S_time
              <<  std::endl;
    std::cout << "--------------------------\n";
    
    output << batch_size << "," << block_size << "," << rank << "," << time
           << "," << gflops << "," << mk_time << "," << packA_V_time
           << "," << packB_S_time << "," << packB_U_time
           << "," << packA_S_time << ","
           << tnum << "," << B_skinny << "," << B_small << std::endl;

    output.close();
  }

  // free_data(A_V, A_V_sim);
  // free_data(A_S, A_S_sim);
  // free_data(B_S, B_S_sim);
  
  return 0;
}
