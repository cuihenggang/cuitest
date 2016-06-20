#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <tbb/tick_count.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

extern "C" {
#include <cblas.h>
}

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;

int main(int argc, char* argv[]) {
  size_t count = 21504 * 1000;
  size_t size = count * sizeof(float);
  void *cpu_ptr;
  void *cpu_ptr2;
  void *gpu_ptr;
  void *gpu_ptr2;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  tbb::tick_count tick_start;

  tick_start = tbb::tick_count::now();
  // cpu_ptr = malloc(size);
  cudaMallocHost(&cpu_ptr, size);
  float *cpu_float_ptr = reinterpret_cast<float *>(cpu_ptr);
  double cpu_malloc_time = (tbb::tick_count::now() - tick_start).seconds();

  for (size_t i = 0; i < count; i++) {
    cpu_float_ptr[i] = i;
  }

  tick_start = tbb::tick_count::now();
  cudaMalloc(&gpu_ptr, size);
  float *gpu_float_ptr = reinterpret_cast<float *>(gpu_ptr);
  double gpu_malloc_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault);
  double cpu_to_gpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  // cpu_ptr2 = malloc(size);
  cudaMallocHost(&cpu_ptr2, size);
  float *cpu_float_ptr2 = reinterpret_cast<float *>(cpu_ptr2);
  tick_start = tbb::tick_count::now();
  memcpy(cpu_ptr2, cpu_ptr, size);
  double cpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cudaMalloc(&gpu_ptr2, size);
  float *gpu_float_ptr2 = reinterpret_cast<float *>(gpu_ptr2);
  tick_start = tbb::tick_count::now();
  cudaMemcpy(gpu_ptr2, gpu_ptr, size, cudaMemcpyDefault);
  double gpu_to_gpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDefault);
  double gpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  for (size_t i = 0; i < count; i++) {
    cpu_float_ptr[i] += cpu_float_ptr2[i];
  }
  double cpu_add_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cblas_saxpy(count, 1, cpu_float_ptr2, 1, cpu_float_ptr, 1);
  double cpu_axpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  float alpha = 1;
  cublasSaxpy(cublas_handle, count, &alpha, cpu_float_ptr2, 1, cpu_float_ptr, 1);
  double gpu_axpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cout << "cpu_malloc_time = " << cpu_malloc_time << endl;
  cout << "gpu_malloc_time = " << gpu_malloc_time << endl;
  cout << "cpu_to_gpu_memcpy_time = " << cpu_to_gpu_memcpy_time << endl;
  cout << "cpu_to_cpu_memcpy_time = " << cpu_to_cpu_memcpy_time << endl;
  cout << "gpu_to_gpu_memcpy_time = " << gpu_to_gpu_memcpy_time << endl;
  cout << "gpu_to_cpu_memcpy_time = " << gpu_to_cpu_memcpy_time << endl;
  cout << "cpu_add_time = " << cpu_add_time << endl;
  cout << "cpu_axpy_time = " << cpu_axpy_time << endl;
  cout << "gpu_axpy_time = " << gpu_axpy_time << endl;
}
