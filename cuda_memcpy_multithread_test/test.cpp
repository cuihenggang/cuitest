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

static void *thread_run(void *arg) {
  size_t thread_id = static_cast<size_t>((unsigned long)(arg));
  size_t local_size = size / num_threads;
  size_t start = local_size * thread_id;
  // void *local_cpu_ptr = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr)[start]);
  // void *local_cpu_ptr2 = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr2)[start]);
  // for (size_t r = 0; r < rounds; r++) {
    // memcpy(local_cpu_ptr2, local_cpu_ptr, local_size);
  // }
  cudaStream_t stream1;
  cudaError_t result;
  result = cudaStreamCreate(&stream1);
  void *local_cpu_ptr = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr)[start]);
  void *local_gpu_ptr = reinterpret_cast<void *>(&reinterpret_cast<char *>(gpu_ptr)[start]);
  for (size_t r = 0; r < rounds; r++) {
    cudaMemcpyAsync(local_gpu_ptr, local_cpu_ptr, local_size, cudaMemcpyDefault, stream1);
  }
  cudaStreamSynchronize(stream1);
  result = cudaStreamDestroy(stream1);
}

int main(int argc, char* argv[]) {
  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  tbb::tick_count tick_start;

  // cpu_ptr = malloc(size);
  // cpu_ptr2 = malloc(size);
  cudaMallocHost(&cpu_ptr, size);
  cudaMallocHost(&cpu_ptr2, size);
  cudaMalloc(&gpu_ptr, size);
  cudaMalloc(&gpu_ptr2, size);

  pthread_t *thread_ids = new pthread_t[num_threads];
  pthread_attr_t thread_attr;
  void *res;
  pthread_attr_init(&thread_attr);

  tick_start = tbb::tick_count::now();
  for (size_t i = 0; i < num_threads; i++) {
    void *thread_arg = (void *)(static_cast<unsigned long>(i));
    pthread_create(&thread_ids[i], &thread_attr, thread_run, thread_arg);
  }
  for (size_t i = 0; i < num_threads; i++) {
    pthread_join(thread_ids[i], &res);
  }
  double cpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();

  cout << "cpu_to_cpu_memcpy_time = " << cpu_to_cpu_memcpy_time << endl;
}
