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
size_t count = 1024 * 1024;
size_t size = count * sizeof(float);
size_t rounds = 100000;
void *cpu_ptr;
void *cpu_ptr2;
void *gpu_ptr;
void *gpu_ptr2;

static void *thread_run(void *arg) {
  size_t thread_id = static_cast<size_t>((unsigned long)(arg));
  // void *local_cpu_ptr = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr)[start]);
  // void *local_cpu_ptr2 = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr2)[start]);
  // for (size_t r = 0; r < rounds; r++) {
    // memcpy(local_cpu_ptr2, local_cpu_ptr, local_size);
  // }
  cudaStream_t stream;
  cudaError_t result;
  result = cudaStreamCreate(&stream);
  tbb::tick_count tick_start = tbb::tick_count::now();
  for (size_t r = 0; r < rounds; r++) {
    if (thread_id == 0) {
      cudaMemcpyAsync(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault, stream);
      cudaStreamSynchronize(stream);
      if ((r + 1) % 1000 == 0) {
        cout << 4 * r << " MB memory CPU->GPU copied in "
           << (tbb::tick_count::now() - tick_start).seconds() << endl;
      }
    } else {
      cudaMemcpyAsync(cpu_ptr2, gpu_ptr2, size, cudaMemcpyDefault, stream);
      cudaStreamSynchronize(stream);
      if ((r + 1) % 1000 == 0) {
        cout << 4 * r << " MB memory GPU->CPU copied in "
            << (tbb::tick_count::now() - tick_start).seconds() << endl;
      }
    }
  }
  result = cudaStreamDestroy(stream);
}

int main(int argc, char* argv[]) {
  cudaMallocHost(&cpu_ptr, size);
  cudaMallocHost(&cpu_ptr2, size);
  cudaMalloc(&gpu_ptr, size);
  cudaMalloc(&gpu_ptr2, size);

  thread_run(static_cast<size_t>(0));
}
