#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <tbb/tick_count.h>

#include "row-op-util.hpp"
#include "gpu-util/syncedmem.hpp"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;
using caffe::SyncedMemory;

int main(int argc, char* argv[]) {
  size_t count = 884832;
  if (argc > 1) {
    count = atoi(argv[1]);
  }
  size_t mem_size = count * sizeof(RowData);
  SyncedMemory *src = new SyncedMemory(mem_size);
  RowData *src_cpu_rows = reinterpret_cast<RowData *>(src->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      src_cpu_rows[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  const RowData *src_gpu_rows = reinterpret_cast<const RowData *>(src->gpu_data());
  void *cpu_buffer;
  CUDA_CHECK(cudaMallocHost(&cpu_buffer, mem_size));
  RowData *cpu_buffer_rows = reinterpret_cast<RowData *>(cpu_buffer);
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      cpu_buffer_rows[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  SyncedMemory *dst = new SyncedMemory(mem_size);
  RowData *dst_gpu_rows = reinterpret_cast<RowData *>(dst->mutable_gpu_data());
  size_t index_size = count * sizeof(DoubleIndex);
  SyncedMemory *index = new SyncedMemory(index_size);
  DoubleIndex *index_cpu_data = reinterpret_cast<DoubleIndex *>(index->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    index_cpu_data[i] = i;
  }
  const DoubleIndex *index_gpu_data = reinterpret_cast<const DoubleIndex *>(index->gpu_data());
  cudaStream_t cuda_stream;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));
  tbb::tick_count tick_start;

  tick_start = tbb::tick_count::now();
  assign_rows_to_double_index_gpu(
        dst_gpu_rows, src_gpu_rows, index_gpu_data, count,
        0, ROW_DATA_SIZE, count * ROW_DATA_SIZE, cuda_stream);
  double assign_rows_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "assign_rows_gpu_time = " << assign_rows_gpu_time << endl;

  tick_start = tbb::tick_count::now();
  memcpy(cpu_buffer_rows, src_cpu_rows, mem_size);
  double cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "cpu_memcpy_time = " << cpu_memcpy_time << endl;

  tick_start = tbb::tick_count::now();
  assign_rows_to_double_index_cpu(
        cpu_buffer_rows, src_cpu_rows, index_cpu_data, count,
        0, ROW_DATA_SIZE, count * ROW_DATA_SIZE);
  double assign_rows_cpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "assign_rows_cpu_time = " << assign_rows_cpu_time << endl;

  tick_start = tbb::tick_count::now();
  CUDA_CHECK(cudaMemcpyAsync(dst_gpu_rows, cpu_buffer_rows, mem_size,
        cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  double copy_to_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "copy_to_gpu_time = " << copy_to_gpu_time << endl;

  tick_start = tbb::tick_count::now();
  CUDA_CHECK(cudaMemcpyAsync(dst_gpu_rows, cpu_buffer_rows, mem_size,
        cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  copy_to_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "copy_to_gpu_time = " << copy_to_gpu_time << endl;

  tick_start = tbb::tick_count::now();
  assign_rows_to_double_index_cpu(
        cpu_buffer_rows, src_cpu_rows, index_cpu_data, count,
        0, ROW_DATA_SIZE, count * ROW_DATA_SIZE);
  assign_rows_cpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "assign_rows_cpu_time = " << assign_rows_cpu_time << endl;

  tick_start = tbb::tick_count::now();
  memcpy(cpu_buffer_rows, src_cpu_rows, mem_size);
  cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "cpu_memcpy_time = " << cpu_memcpy_time << endl;

  tick_start = tbb::tick_count::now();
  assign_rows_to_double_index_gpu(
        dst_gpu_rows, src_gpu_rows, index_gpu_data, count,
        0, ROW_DATA_SIZE, count * ROW_DATA_SIZE, cuda_stream);
  assign_rows_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "assign_rows_gpu_time = " << assign_rows_gpu_time << endl;
}
