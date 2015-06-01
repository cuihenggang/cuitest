#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <pthread.h>

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

size_t count = 21504 * 1000;
size_t size = count * sizeof(float);
size_t num_threads = 2;
size_t rounds = 100;
void *cpu_ptr;
void *cpu_ptr2;
void *gpu_ptr;
void *gpu_ptr2;
void *gpu_ptr3;
void *gpu_ptr4;
void *gpu_ptr5;

void do_memcpy() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  tbb::tick_count tick_start = tbb::tick_count::now();
  for (size_t r = 0; r < rounds; r++) {
    cudaMemcpyAsync(gpu_ptr5, gpu_ptr4, size, cudaMemcpyDefault, stream);
  }
  cudaStreamSynchronize(stream);

  double compute_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "memcpy_time = " << compute_time << endl;
  cudaStreamDestroy(stream);
}

void do_compute() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, stream);
  
  float *A = reinterpret_cast<float *>(gpu_ptr);
  float *B = reinterpret_cast<float *>(gpu_ptr2);
  float *C = reinterpret_cast<float *>(gpu_ptr3);
  float alpha = 1;
  float beta = 0;
  int N = 4000;
  int M = 4000;
  int K = 4000;

  tbb::tick_count tick_start = tbb::tick_count::now();
  for (size_t r = 0; r < 10; r++) {
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, K, A, N, &beta, C, N);
  }
  cudaStreamSynchronize(stream);

  double compute_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "compute_time = " << compute_time << endl;
  cudaStreamDestroy(stream);
}

static void *thread_run(void *arg) {
  size_t thread_id = static_cast<size_t>((unsigned long)(arg));
  if (thread_id == 0) {
    do_compute();
  } else {
    do_memcpy();
  }
}

int main(int argc, char* argv[]) {
  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  // cpu_ptr = malloc(size);
  // cpu_ptr2 = malloc(size);
  cudaMallocHost(&cpu_ptr, size);
  cudaMallocHost(&cpu_ptr2, size);
  cudaMalloc(&gpu_ptr, size);
  cudaMalloc(&gpu_ptr2, size);
  cudaMalloc(&gpu_ptr3, size);
  cudaMalloc(&gpu_ptr4, size);
  cudaMalloc(&gpu_ptr5, size);

  float *cpu_float_ptr = reinterpret_cast<float *>(cpu_ptr);
  for (size_t i = 0; i < count; i++) {
    cpu_float_ptr[i] = 1;
  }
  cudaMemcpyAsync(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault);
  cudaMemcpyAsync(gpu_ptr2, cpu_ptr, size, cudaMemcpyDefault);

  pthread_t *thread_ids = new pthread_t[num_threads];
  pthread_attr_t thread_attr;
  void *res;
  pthread_attr_init(&thread_attr);

  for (size_t i = 0; i < num_threads; i++) {
    void *thread_arg = (void *)(static_cast<unsigned long>(i));
    pthread_create(&thread_ids[i], &thread_attr, thread_run, thread_arg);
  }
  for (size_t i = 0; i < num_threads; i++) {
    pthread_join(thread_ids[i], &res);
  }
}
