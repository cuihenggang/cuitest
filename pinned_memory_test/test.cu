#include <iostream>
#include <assert.h>

#include <glog/logging.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/mman.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#if __CUDA_ARCH__ < 200
    int CUDA_ARCH = 100;
#else
    int CUDA_ARCH = 200;
#endif

using namespace std;

int main() {
  cout << "CUDA_ARCH = " << CUDA_ARCH << endl;

  int driverVersion;
  CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
  cout << "driverVersion = " << driverVersion << endl;	

  cudaDeviceProp deviceProp;
  int devID = 0;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
  cout << "deviceProp.major = " << deviceProp.major << endl;
  cout << "deviceProp.minor = " << deviceProp.minor << endl;
  if (((deviceProp.major << 4) + deviceProp.minor) < 0x20) {
    cout << "binomialOptions requires Compute Capability of SM 2.0 or higher to run.\n";
    cudaDeviceReset();
    exit(0);
  }

  size_t count = 0;
  size_t size = 64 * 1024 * 1024 * sizeof(float);
  while (true) {
    void *host_array;
    if (cudaMallocHost(&host_array, size) == cudaSuccess) {
      // memset(host_array, 0, size);
      count++;
      cout << "Allocated " << count * 256 << " MB" << endl;
    } else {
      cout << "Allocation failed at " << count * 256 << " MB" << endl;
    }
  }

  // CUDA_CHECK(cudaHostAlloc(&host_array, size, cudaHostAllocMapped));
  // CUDA_CHECK(cudaMalloc(&host_array, size));
  // CHECK(host_array = malloc(size));
  // CHECK_EQ(mlock(host_array, size), 0);
}