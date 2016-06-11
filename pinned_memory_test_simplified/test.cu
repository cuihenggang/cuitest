#include <iostream>
#include <assert.h>

// #include <glog/logging.h>

#include <cuda.h>
#include <cuda_runtime.h>

// #include <sys/mman.h>

using namespace std;

int main() {
  size_t count = 0;
  size_t size = 64 * 1024 * 1024 * sizeof(float);
  while (true) {
    void *host_array;
    cudaError errono = cudaMallocHost(&host_array, size);
    if (errono == cudaSuccess) {
      count++;
      cout << "Allocated " << count * 256 << " MB" << endl;
    } else {
      cout << "Allocation failed at " << count * 256 << " MB, with errono " << errono << endl;
      exit(1);
    }
    if (count > 20) {
      exit(0);
    }
  }
}
