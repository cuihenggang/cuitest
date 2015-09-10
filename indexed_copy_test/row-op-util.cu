#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <iostream>

#include "row-op-util.hpp"
#include "gpu-util/syncedmem.hpp"
#include "gpu-util/math_functions.hpp"

using std::cout;
using std::endl;

__global__ void assign_rows_to_index_kernel(
    val_t *y, const val_t *x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_id = i / row_size;
    size_t val_id = i % row_size;
    if (index[row_id] >= 0) {
      size_t y_idx = index[row_id] * row_size + val_id;
      if (num_vals_limit < 0 || y_idx < num_vals_limit) {
        y[y_idx] = x[i];
      }
    }
  }
}

void assign_rows_to_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  assign_rows_to_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void assign_rows_from_index_kernel(
    val_t *y, const val_t *x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_id = i / row_size;
    size_t val_id = i % row_size;
    if (index[row_id] >= 0) {
      size_t x_idx = index[row_id] * row_size + val_id;
      if (num_vals_limit < 0 || x_idx < num_vals_limit) {
        y[i] = x[x_idx];
      }
    }
  }
}

void assign_rows_from_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  assign_rows_from_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void add_rows_from_index_kernel(
    val_t *y, const val_t *x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_id = i / row_size;
    size_t val_id = i % row_size;
    if (index[row_id] >= 0) {
      size_t x_idx = index[row_id] * row_size + val_id;
      if (num_vals_limit < 0 || x_idx < num_vals_limit) {
        y[i] += x[x_idx];
      }
    }
  }
}

void add_rows_from_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  add_rows_from_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}


__global__ void assign_rows_to_double_index_kernel(
    val_t *y, const val_t *x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_index_id = i / row_size;
    size_t val_id = i % row_size;
    /* Assign rows from "id1" to "id0" */
    size_t row_from = index[row_index_id].id1 + index_offset.id1;
    size_t row_to = index[row_index_id].id0 + index_offset.id0;
    size_t x_idx = row_from * row_size + val_id;
    size_t y_idx = row_to * row_size + val_id;
    if (num_vals_limit < 0 || y_idx < num_vals_limit) {
      y[y_idx] = x[x_idx];
    }
  }
}

void assign_rows_to_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  assign_rows_to_double_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, index_offset, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void assign_rows_from_double_index_kernel(
    val_t *y, const val_t *x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_index_id = i / row_size;
    size_t val_id = i % row_size;
    /* Assign rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    size_t x_idx = row_from * row_size + val_id;
    size_t y_idx = row_to * row_size + val_id;
    if (num_vals_limit < 0 || x_idx < num_vals_limit) {
      y[y_idx] = x[x_idx];
    }
  }
}

void assign_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  assign_rows_from_double_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, index_offset, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void add_rows_from_double_index_kernel(
    val_t *y, const val_t *x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  CUDA_KERNEL_LOOP(i, num_rows * row_size) {
    size_t row_index_id = i / row_size;
    size_t val_id = i % row_size;
    /* Add rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    size_t x_idx = row_from * row_size + val_id;
    size_t y_idx = row_to * row_size + val_id;
    if (num_vals_limit < 0 || x_idx < num_vals_limit) {
      y[y_idx] += x[x_idx];
    }
  }
}

void add_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream) {
  if (num_rows == 0) {
    return;
  }
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  add_rows_from_double_index_kernel
      <<<caffe::CAFFE_GET_BLOCKS(num_rows * row_size),
         caffe::CAFFE_CUDA_NUM_THREADS, 0, cuda_stream>>>
      (y, x, index, num_rows, index_offset, row_size, num_vals_limit);
  CUDA_POST_KERNEL_CHECK;
}
