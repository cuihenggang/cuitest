#ifndef GPU_UTIL_MATH_UTIL_HPP_
#define GPU_UTIL_MATH_UTIL_HPP_

#include "math_functions.hpp"

template <typename Dtype>
void gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    int m, int n, int k,
    const Dtype alpha, const Dtype *a, int lda, const Dtype *b, int ldb,
    const Dtype beta, Dtype *c, int ldc,
    int flag) {
  if (flag) {
    caffe::caffe_gpu_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb,
        beta, c, ldc);
  } else {
    caffe::caffe_cpu_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb,
        beta, c, ldc);
  }
}

template <typename Dtype>
Dtype asum(int n, const float *x, int flag) {
  if (flag) {
    return caffe::caffe_gpu_asum(n, x);
  } else {
    return caffe::caffe_cpu_asum(n, x);
  }
}

template <typename Dtype>
Dtype dot(int n, const float *x, const float *y, int flag) {
  if (flag) {
    return caffe::caffe_gpu_dot(n, x, y);
  } else {
    return caffe::caffe_cpu_dot(n, x, y);
  }
}

#endif  // GPU_UTIL_MATH_UTIL_HPP_
