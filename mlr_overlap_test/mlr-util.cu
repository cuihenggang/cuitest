#if !defined(CPU_ONLY)

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "syncedmem.hpp"
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

__global__ void SoftmaxBatchAndAdjust_kernel(
    size_t n, size_t size, float *vecs, const uint *labels) {
  CUDA_KERNEL_LOOP(i, n) {
    float *vec = &vecs[i * size];
    Softmax_device(vec, size);
    vec[labels[i]] -= 1.; // See Bishop PRML (2006) Eq. (4.109)
  }
}

void SoftmaxBatchAndAdjust_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, float *vecs, const uint *labels) {
  SoftmaxBatchAndAdjust_kernel
      <<<caffe::CAFFE_GET_BLOCKS(n), caffe::CAFFE_CUDA_NUM_THREADS,
         0, cuda_stream>>>
      (n, size, vecs, labels);
}

__global__ void SoftmaxBatchAndEntropyLoss_kernel(
    size_t n, size_t size, float *vecs, const uint *labels, float *losses) {
  CUDA_KERNEL_LOOP(i, n) {
    float *vec = &vecs[i * size];
    Softmax_device(vec, size);
    losses[i] = -logf(vec[labels[i]]);
  }
}

void SoftmaxBatchAndEntropyLoss_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, float *vecs, const uint *labels, float *losses) {
  SoftmaxBatchAndEntropyLoss_kernel
      <<<caffe::CAFFE_GET_BLOCKS(n), caffe::CAFFE_CUDA_NUM_THREADS,
         0, cuda_stream>>>
      (n, size, vecs, labels, losses);
}

__device__ float ZeroOneLoss(size_t size, float *vec, uint label) {
  uint max_idx = 0;
  float max_val = vec[0];
  for (uint i = 1; i < size; i++) {
    if (vec[i] > max_val) {
      max_val = vec[i];
      max_idx = i;
    }
  }
  return (max_idx == label) ? 0 : 1;
}

__global__ void SoftmaxBatchAndZeroOneLoss_kernel(
    size_t n, size_t size, float *vecs, const uint *labels, float *losses) {
  CUDA_KERNEL_LOOP(i, n) {
    float *vec = &vecs[i * size];
    Softmax_device(vec, size);
    losses[i] = ZeroOneLoss(size, vec, labels[i]);
  }
}

void SoftmaxBatchAndZeroOneLoss_gpu(
    cudaStream_t cuda_stream,
    size_t n, size_t size, float *vecs, const uint *labels, float *losses) {
  SoftmaxBatchAndZeroOneLoss_kernel
      <<<caffe::CAFFE_GET_BLOCKS(n), caffe::CAFFE_CUDA_NUM_THREADS,
         0, cuda_stream>>>
      (n, size, vecs, labels, losses);
}

__global__ void empty_kernel() {

}

void empty_gpu_func() {
  empty_kernel<<<1, 1>>>();
}

#endif