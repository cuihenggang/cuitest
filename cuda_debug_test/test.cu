#include <iostream>
#include <assert.h>

#include <glog/logging.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

using namespace std;

int main() {
  size_t count = 200 * 1000 * 1000;
  size_t size = count * sizeof(float);
  void *arrays[10];
  for (size_t i = 0; i < 10; i++) {
    CUDA_CHECK(cudaMalloc(&arrays[i], size));
  }
  void *host_array;
  CUDA_CHECK(cudaMallocHost(&host_array, size));
  for (size_t i = 0; i < 10; i++) {
    CUDA_CHECK(cudaMemset(arrays[8], 0, size));
    cout << "memset" << i << endl;
    for (size_t j = 0; j < 10; j++) {
      void *array = arrays[j];
      CUDA_CHECK(cudaMemcpy(host_array, array, size, cudaMemcpyDeviceToHost));
      float *floats = reinterpret_cast<float *>(host_array);
      float sum = 0.0;
      for (size_t i = 0; i < count; i++) {
        sum += floats[i];
      }
      cout << "sum" << j << "=" << sum << endl;
    }
  }
  cout << "sum" << "=" << sum << endl;
}