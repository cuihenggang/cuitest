#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <tbb/tick_count.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;

int main(int argc, char* argv[]) {
  size_t size = 21504 * 1000 * 4;
  void *cpu_ptr;
  void *cpu_ptr2;
  void *gpu_ptr;
  void *gpu_ptr2;
  tbb::tick_count tick_start;

  tick_start = tbb::tick_count::now();
  cpu_ptr = new char[size];
  double cpu_malloc_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMalloc(&gpu_ptr, size);
  double gpu_malloc_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault);
  double cpu_to_gpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cpu_ptr2 = new char[size];
  tick_start = tbb::tick_count::now();
  memcpy(cpu_ptr2, cpu_ptr, size);
  double cpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cudaMalloc(&gpu_ptr2, size);
  tick_start = tbb::tick_count::now();
  cudaMemcpy(gpu_ptr2, gpu_ptr, size, cudaMemcpyDefault);
  double gpu_to_gpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDefault);
  double gpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cout << "cpu_malloc_time = " << cpu_malloc_time << endl;
  cout << "gpu_malloc_time = " << gpu_malloc_time << endl;
  cout << "cpu_to_gpu_memcpy_time = " << cpu_to_gpu_memcpy_time << endl;
  cout << "cpu_to_cpu_memcpy_time = " << cpu_to_cpu_memcpy_time << endl;
  cout << "gpu_to_gpu_memcpy_time = " << gpu_to_gpu_memcpy_time << endl;
  cout << "gpu_to_cpu_memcpy_time = " << gpu_to_cpu_memcpy_time << endl;
}
