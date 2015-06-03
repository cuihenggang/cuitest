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

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

__global__ void copy_kernel(
    float *dst, float *src, size_t n) {
  CUDA_KERNEL_LOOP(i, n) {
    dst[i] = src[i];
  }
}

__global__ void index_kernel(
    float *dst, float *src, size_t *index, size_t n) {
  CUDA_KERNEL_LOOP(i, n) {
    dst[i] = src[index[i]];
  }
}

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;

int main(int argc, char* argv[]) {
  size_t count = 21504 * 1000;
  size_t float_size = count * sizeof(float);
  size_t index_size = count * sizeof(size_t);
  float *cpu_ptr;
  float *cpu_ptr2;
  float *gpu_ptr;
  float *gpu_ptr2;
  size_t *cpu_index_ptr;
  size_t *gpu_index_ptr;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  tbb::tick_count tick_start;

  cpu_ptr = reinterpret_cast<float *>(malloc(float_size));
  cpu_ptr2 = reinterpret_cast<float *>(malloc(float_size));
  cudaMalloc(&gpu_ptr, float_size);
  cudaMalloc(&gpu_ptr2, float_size);
  cpu_index_ptr = reinterpret_cast<size_t *>(malloc(index_size));
  cudaMalloc(&gpu_index_ptr, index_size);

  for (size_t i = 0; i < count; i++) {
    cpu_ptr[i] = i;
    cpu_ptr2[i] = i + 1;
    cpu_index_ptr[i] = (i % 10) * (count / 10) + i / 10;
  }
  cudaMemcpy(gpu_ptr, cpu_ptr, float_size, cudaMemcpyDefault);
  cudaMemcpy(gpu_index_ptr, cpu_index_ptr, index_size, cudaMemcpyDefault);

  tick_start = tbb::tick_count::now();
  for (size_t i = 0; i < count; i++) {
    cpu_ptr2[i] = cpu_ptr[cpu_index_ptr[i]];
  }
  double cpu_index_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  index_kernel
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (gpu_ptr2, gpu_ptr, gpu_index_ptr, count);
  double gpu_index_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  for (size_t i = 0; i < count; i++) {
    cpu_ptr2[i] = cpu_ptr[i];
  }
  double cpu_assign_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  memcpy(cpu_ptr2, cpu_ptr, float_size);
  double cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  cudaMemcpy(gpu_ptr2, gpu_ptr, float_size, cudaMemcpyDefault);
  double gpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  tick_start = tbb::tick_count::now();
  copy_kernel
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (gpu_ptr2, gpu_ptr, count);
  double gpu_assign_time = (tbb::tick_count::now() - tick_start).seconds();

  cout << "cpu_memcpy_time = " << cpu_memcpy_time << endl;
  cout << "cpu_assign_time = " << cpu_assign_time << endl;
  cout << "gpu_memcpy_time = " << gpu_memcpy_time << endl;
  cout << "gpu_assign_time = " << gpu_assign_time << endl;
  cout << "cpu_index_time = " << cpu_index_time << endl;
  cout << "gpu_index_time = " << gpu_index_time << endl;
}
