#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
  size_t count = 20 * 1000 * 1000;
  size_t size = count * sizeof(float);
  void *arrays[10];
  for (size_t i = 0; i < 10; i++) {
    assert(cudaMalloc(&arrays[i], size) == cudaSuccess);
  }
  void *host_array;
  assert(cudaMallocHost(&host_array, size) == cudaSuccess);
  for (size_t i = 0; i < 10; i++) {
    assert(cudaMemset(arrays[8], 0, size) == cudaSuccess);
    cout << "memset" << i << endl;
    for (size_t j = 0; j < 10; j++) {
      void *array = arrays[j];
      assert(cudaMemcpy(host_array, array, size, cudaMemcpyDeviceToHost) == cudaSuccess);
      float *floats = reinterpret_cast<float *>(host_array);
      float sum = 0.0;
      for (size_t i = 0; i < count; i++) {
        sum += floats[i];
      }
      cout << "sum" << j << "=" << sum << endl;
    }
  }
}