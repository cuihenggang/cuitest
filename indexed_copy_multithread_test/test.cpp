#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <pthread.h>

#include <tbb/tick_count.h>

#include "row-op-util.hpp"
#include "gpu-util/syncedmem.hpp"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;
using caffe::SyncedMemory;

size_t count;
int rounds;
int mode;
size_t mem_size0;
size_t mem_size1;
SyncedMemory *src0;
SyncedMemory *src1;
RowData *src_cpu_rows0;
RowData *src_cpu_rows1;
const RowData *src_gpu_rows1;
void *cpu_buffer0;
void *cpu_buffer1;
RowData *cpu_buffer_rows0;
RowData *cpu_buffer_rows1;
SyncedMemory *dst0;
SyncedMemory *dst1;
RowData *dst_gpu_rows0;
RowData *dst_cpu_rows1;
size_t index_size0;
size_t index_size1;
SyncedMemory *index_mem0;
SyncedMemory *index_mem1;
DoubleIndex *index_cpu_data0;
DoubleIndex *index_cpu_data1;
cudaStream_t cuda_stream0;
cudaStream_t cuda_stream1;

void init_cpu_to_gpu() {
  mem_size0 = count * sizeof(RowData);
  src0 = new SyncedMemory(mem_size0);
  src_cpu_rows0 = reinterpret_cast<RowData *>(src0->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      src_cpu_rows0[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  CUDA_CHECK(cudaMallocHost(&cpu_buffer0, mem_size0));
  cpu_buffer_rows0 = reinterpret_cast<RowData *>(cpu_buffer0);
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      cpu_buffer_rows0[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  dst0 = new SyncedMemory(mem_size0);
  dst_gpu_rows0 = reinterpret_cast<RowData *>(dst0->mutable_gpu_data());
  index_size0 = count * sizeof(DoubleIndex);
  index_mem0 = new SyncedMemory(index_size0);
  index_cpu_data0 = reinterpret_cast<DoubleIndex *>(index_mem0->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    index_cpu_data0[i] = i;
  }
  CUDA_CHECK(cudaStreamCreate(&cuda_stream0));
}

void cpu_to_gpu() {
  tbb::tick_count tick_start;

  tick_start = tbb::tick_count::now();
  memcpy(cpu_buffer_rows0, src_cpu_rows0, mem_size0);
  double copy_to_buffer_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "memcpy_to_buffer_time0 = " << copy_to_buffer_time << endl;

  tick_start = tbb::tick_count::now();
  CUDA_CHECK(cudaMemcpyAsync(dst_gpu_rows0, cpu_buffer_rows0, mem_size0,
        cudaMemcpyDefault, cuda_stream0));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream0));
  double copy_to_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "cudamemcpy_to_gpu_time0 = " << copy_to_gpu_time << endl;
}

void init_gpu_to_cpu() {
  mem_size1 = count * sizeof(RowData);
  src1 = new SyncedMemory(mem_size1);
  src_cpu_rows1 = reinterpret_cast<RowData *>(src1->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      src_cpu_rows1[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  src_gpu_rows1 = reinterpret_cast<const RowData *>(src1->gpu_data());
  CUDA_CHECK(cudaMallocHost(&cpu_buffer1, mem_size1));
  cpu_buffer_rows1 = reinterpret_cast<RowData *>(cpu_buffer1);
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < ROW_DATA_SIZE; j++) {
      cpu_buffer_rows1[i].data[j] = i * ROW_DATA_SIZE + j;
    }
  }
  dst1 = new SyncedMemory(mem_size1);
  dst_cpu_rows1 = reinterpret_cast<RowData *>(dst1->mutable_cpu_data());
  index_size1 = count * sizeof(DoubleIndex);
  index_mem1 = new SyncedMemory(index_size1);
  index_cpu_data1 = reinterpret_cast<DoubleIndex *>(index_mem1->mutable_cpu_data());
  for (int i = 0; i < count; i++) {
    index_cpu_data1[i] = i;
  }
  CUDA_CHECK(cudaStreamCreate(&cuda_stream1));  
}

void gpu_to_cpu() {
  tbb::tick_count tick_start;

  tick_start = tbb::tick_count::now();
  CUDA_CHECK(cudaMemcpyAsync(cpu_buffer_rows1, src_gpu_rows1, mem_size1,
        cudaMemcpyDefault, cuda_stream1));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream1));
  double copy_to_buffer_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "cudamemcpy_to_buffer_time1 = " << copy_to_buffer_time << endl;

  tick_start = tbb::tick_count::now();
  memcpy(dst_cpu_rows1, cpu_buffer_rows1, mem_size1);
  double copy_to_cpu_time = (tbb::tick_count::now() - tick_start).seconds();
  cout << "memcpy_to_cpu_time1 = " << copy_to_cpu_time << endl;
}

static void *thread_run(void *arg) {
  size_t thread_id = static_cast<size_t>((unsigned long)(arg));

  if ((mode & 0x1) && thread_id == 0) {
    init_cpu_to_gpu();
    tbb::tick_count tick_start = tbb::tick_count::now();
    for (int i = 0; i < rounds; i++) {
      cpu_to_gpu();
    }
    double cpu_to_gpu_time = (tbb::tick_count::now() - tick_start).seconds();
    cout << "cpu_to_gpu_time0 = " << cpu_to_gpu_time << endl;
  }
  if ((mode & 0x2) && thread_id == 1) {
    init_gpu_to_cpu();
    tbb::tick_count tick_start = tbb::tick_count::now();
    for (int i = 0; i < rounds; i++) {
      gpu_to_cpu();
    }
    double gpu_to_cpu_time = (tbb::tick_count::now() - tick_start).seconds();
    cout << "gpu_to_cpu_time1 = " << gpu_to_cpu_time << endl;
  }
}

int main(int argc, char* argv[]) {
  count = 884832;
  if (argc > 1) {
    count = atoi(argv[1]);
  }
  rounds = 10;
  mode = 3;
  if (argc > 2) {
    mode = atoi(argv[2]);
  }
  // init_cpu_to_gpu();
  // cpu_to_gpu();

  int num_threads = 2;
  pthread_t *thread_ids = new pthread_t[num_threads];
  pthread_attr_t thread_attr;
  void *res;
  pthread_attr_init(&thread_attr);

  tbb::tick_count tick_start;
  tick_start = tbb::tick_count::now();
  for (size_t i = 0; i < num_threads; i++) {
    void *thread_arg = (void *)(static_cast<unsigned long>(i));
    pthread_create(&thread_ids[i], &thread_attr, thread_run, thread_arg);
  }
  for (size_t i = 0; i < num_threads; i++) {
    pthread_join(thread_ids[i], &res);
  }
  double cpu_to_cpu_memcpy_time = (tbb::tick_count::now() - tick_start).seconds();
}
