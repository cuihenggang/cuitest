#if !defined(CPU_ONLY)

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
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
    float *vec, const size_t size, uint label) {
  Softmax_device(vec, size);
  vec[label] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
}

void SoftmaxAndAdjust_gpu(float *vec, const size_t size, uint label) {
  SoftmaxAndAdjust_kernel<<<1, 1>>>(vec, size, label);
}

#endif