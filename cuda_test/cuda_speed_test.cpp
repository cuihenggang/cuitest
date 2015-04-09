#include <iostream>
#include <string>
#include <vector>

#include <tbb/tick_count.h>

#include "syncedmem.hpp"
#include "math_functions.hpp"

using namespace std;
using namespace caffe;

typedef unsigned int uint;

void gpu_add(uint data_size, int skip) {
  SyncedMemory a(data_size * sizeof(float));
  SyncedMemory b(data_size * sizeof(float));
  float *a_cptr = static_cast<float *>(a.mutable_cpu_data());
  float *b_cptr = static_cast<float *>(b.mutable_cpu_data());
  if (skip > 3) {
    return;
  }
  for (uint i = 0; i < data_size; i++) {
    a_cptr[i] = 2;
    b_cptr[i] = 3;
  }
  if (skip > 2) {
    return;
  }
  const float *a_gptr = static_cast<const float *>(a.gpu_data());
  float *b_gptr = static_cast<float *>(b.mutable_gpu_data());
  if (skip > 1) {
    return;
  }
  // caffe_gpu_add(data_size, a_gptr, b_gptr, b_gptr);
  caffe_gpu_axpy<float>(data_size, 0.01, a_gptr, b_gptr);
  if (skip > 0) {
    return;
  }
  b_cptr = static_cast<float *>(b.mutable_cpu_data());
}

void gpu_add_test(uint rounds, uint data_size, int skip) {
  for (uint r = 0; r < rounds; r++) {
    gpu_add(data_size, skip);
  }
}

void cpu_add(uint data_size, int skip) {
  SyncedMemory a(data_size * sizeof(float));
  SyncedMemory b(data_size * sizeof(float));
  float *a_cptr = static_cast<float *>(a.mutable_cpu_data());
  float *b_cptr = static_cast<float *>(b.mutable_cpu_data());
  if (skip > 3) {
    return;
  }
  for (uint i = 0; i < data_size; i++) {
    a_cptr[i] = 2;
    b_cptr[i] = 3;
  }
  if (skip > 2) {
    return;
  }
  for (uint i = 0; i < data_size; i++) {
    b_cptr[i] = a_cptr[i] * 0.01 + b_cptr[i];
  }
}

void cpu_add_test(uint rounds, uint data_size, int skip) {
  for (uint r = 0; r < rounds; r++) {
    cpu_add(data_size, skip);
  }
}

void gpu_data_ready_add_test(uint rounds, uint data_size, int skip) {
  SyncedMemory a(data_size * sizeof(float));
  SyncedMemory b(data_size * sizeof(float));
  float *a_cptr = static_cast<float *>(a.mutable_cpu_data());
  float *b_cptr = static_cast<float *>(b.mutable_cpu_data());
  for (uint i = 0; i < data_size; i++) {
    a_cptr[i] = 2;
    b_cptr[i] = 3;
  }
  const float *a_gptr = static_cast<const float *>(a.gpu_data());
  float *b_gptr = static_cast<float *>(b.mutable_gpu_data());
  for (uint r = 0; r < rounds; r++) {
    // caffe_gpu_add(data_size, a_gptr, b_gptr, b_gptr);
    caffe_gpu_axpy<float>(data_size, 0.01, a_gptr, b_gptr);
  }
  b_cptr = static_cast<float *>(b.mutable_cpu_data());
}

void cpu_data_ready_add_test(uint rounds, uint data_size, int skip) {
  SyncedMemory a(data_size * sizeof(float));
  SyncedMemory b(data_size * sizeof(float));
  float *a_cptr = static_cast<float *>(a.mutable_cpu_data());
  float *b_cptr = static_cast<float *>(b.mutable_cpu_data());
  for (uint i = 0; i < data_size; i++) {
    a_cptr[i] = 2;
    b_cptr[i] = 3;
  }
  for (uint r = 0; r < rounds; r++) {
    for (uint i = 0; i < data_size; i++) {
      // b_cptr[i] = a_cptr[i] + b_cptr[i];
      b_cptr[i] = a_cptr[i] * 0.01 + b_cptr[i];
    }
  }
}

int main(int argc, char* argv[])
{
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;

  Caffe::SetDevice(0);
  uint data_size = 10000;
  uint rounds = 1000;
  uint mode = 0;
  int skip = 0;

  if (argc > 1) {
    data_size = atoi(argv[1]);
  }
  if (argc > 2) {
    rounds = atoi(argv[2]);
  }
  if (argc > 3) {
    mode = atoi(argv[3]);
  }
  if (argc > 4) {
    skip = atoi(argv[4]);
  }

  tick_start = tbb::tick_count::now();
  switch (mode) {
    case 0:
      gpu_add_test(rounds, data_size, skip);
      break;
    case 1:
      cpu_add_test(rounds, data_size, skip);
      break;
    case 2:
      gpu_data_ready_add_test(rounds, data_size, skip);
      break;
    case 3:
      cpu_data_ready_add_test(rounds, data_size, skip);
      break;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "total_time = " << total_time << std::endl;
}
