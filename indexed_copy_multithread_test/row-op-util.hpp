#ifndef __row_op_util_hpp__
#define __row_op_util_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Overloaded functions to handle operations on RowData or RowOpVal
// each of which can be of the vector type or unordered map

#include <vector>

#include "gpu-util/math_functions.hpp"


#include <boost/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include <string>
#include <vector>
#include <utility>

typedef uint8_t command_t;
typedef uint32_t row_idx_t;
typedef uint32_t col_idx_t;
typedef float val_t;
typedef uint32_t table_id_t;
typedef int iter_t;

typedef std::pair<table_id_t, row_idx_t> TableRow;
typedef struct {
  table_id_t table;
  row_idx_t row;
} table_row_t;

typedef boost::unordered_map<col_idx_t, val_t> UMapData;
typedef std::vector<val_t> VectorData;
// #define ROW_DATA_SIZE 1000
// #define ROW_DATA_SIZE 21504
// #define ROW_DATA_SIZE 16
#define ROW_DATA_SIZE 128
// #define ROW_DATA_SIZE 512
// #define ROW_DATA_SIZE 10
struct ArrayData {
  val_t data[ROW_DATA_SIZE];
  void init() {
    for (uint32_t i = 0; i < ROW_DATA_SIZE; i++) {
      data[i] = 0;
    }
  }
  ArrayData() {
    init();
  }
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & data;
  }
};

typedef ArrayData RowData;
typedef ArrayData RowOpVal;

struct DoubleIndex {
  int id0;
  int id1;
  DoubleIndex(int id0_i = 0, int id1_i = 0) : id0(id0_i), id1(id1_i) {}
};

void operator += (ArrayData& left, const ArrayData& right);
void add_row_batch(
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size);
void add_row_batch(
    cublasHandle_t cublas_handle,
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size);
void assign_rows_to_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream);
void assign_rows_from_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream);
void add_rows_from_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const int *index,
    size_t num_rows, size_t row_size, int num_vals_limit,
    cudaStream_t cuda_stream);
void assign_rows_to_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream);
void assign_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream);
void add_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit,
    cudaStream_t cuda_stream);

inline void operator += (ArrayData& left, const ArrayData& right) {
  for (uint i = 0; i < ROW_DATA_SIZE; i++) {
    left.data[i] += right.data[i];
  }
}

inline void add_row_batch(
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  size_t n = batch_size * ROW_DATA_SIZE;
  caffe::caffe_axpy<val_t>(n, 1, x, y);
}

inline void add_row_batch_gpu(
    cublasHandle_t cublas_handle,
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  size_t n = batch_size * ROW_DATA_SIZE;
  caffe::caffe_gpu_axpy<val_t>(cublas_handle, n, 1, x, y);
}

inline void assign_rows_to_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Assign rows from "id1" to "id0" */
    size_t row_from = index[row_index_id].id1 + index_offset.id1;
    size_t row_to = index[row_index_id].id0 + index_offset.id0;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (num_vals_limit < 0 || y_idx < static_cast<size_t>(num_vals_limit)) {
        y[y_idx] = x[x_idx];
      }
    }
  }
}

inline void assign_rows_from_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Add rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (num_vals_limit < 0 || x_idx < static_cast<size_t>(num_vals_limit)) {
        y[y_idx] = x[x_idx];
      }
    }
  }      
}

inline void add_rows_from_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    int num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Add rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (num_vals_limit < 0 || x_idx < static_cast<size_t>(num_vals_limit)) {
        y[y_idx] += x[x_idx];
      }
    }
  }      
}

#endif  // defined __row_op_util_hpp__
