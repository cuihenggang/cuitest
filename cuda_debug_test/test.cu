#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
  size_t count = 200 * 1000 * 1000;
  size_t size = count * sizeof(float);
  void *array;
  assert(cudaMalloc(&array, size) == cudaSuccess);
  void *host_array;
  assert(cudaMallocHost(&host_array, size) == cudaSuccess);
  assert(cudaMemset(array, 0, size) == cudaSuccess);
  assert(cudaMemcpy(host_array, array, size, cudaMemcpyDeviceToHost) == cudaSuccess);
  float *floats = reinterpret_cast<float *>(host_array);
  float sum = 0.0;
  for (size_t i = 0; i < count; i++) {
    sum += floats[i];
  }
  cout << "sum" << "=" << sum << endl;
}