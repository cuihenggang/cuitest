#if !defined(CPU_ONLY)

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "math_functions.hpp"

#define kCutoff 1e-15

__device__ float LogSum_device(float log_a, float log_b) {
  return (log_a < log_b) ? log_b + logf(1 + expf(log_a - log_b)) :
                           log_a + logf(1 + expf(log_b - log_a));
}

__device__ float LogSumVec_device(const float *logvec, size_t size) {
	float sum = 0.;
	sum = logvec[0];
	for (uint i = 1; i < size; ++i) {
		sum = LogSum_device(sum, logvec[i]);
	}
	return sum;
}

__device__ void Softmax_device(float *vec, size_t size) {
  // TODO(wdai): Figure out why this is necessary. Doubt it is.
	for (uint i = 0; i < size; ++i) {
		if (abs(vec[i]) < kCutoff) {
			vec[i] = kCutoff;
    }
	}
	float lsum = LogSumVec_device(vec, size);
	for (uint i = 0; i < size; ++i) {
		vec[i] = expf(vec[i] - lsum);
		//(*vec)[i] = exp((*vec)[i] - lsum);
    vec[i] = vec[i] > 1 ? 1. : vec[i];
  }
}

__global__ void SoftmaxAndAdjust_kernel(
    size_t n, size_t size, float *vecs, uint *labels) {
  CUDA_KERNEL_LOOP(index, n) {
    float *vec = &vecs[index * size];
    Softmax_device(vec, size);
    vec[labels[index]] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
  }
}

__global__ void empty_kernel() {

}

void SoftmaxAndAdjust_gpu(size_t n, size_t size, float *vecs, uint *labels) {
  SoftmaxAndAdjust_kernel
      <<<caffe::CAFFE_GET_BLOCKS(n), caffe::CAFFE_CUDA_NUM_THREADS>>>
      (n, size, vecs, labels);
}

void empty_gpu_func() {
  empty_kernel<<<1, 1>>>();
}

#endif