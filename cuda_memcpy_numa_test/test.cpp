#include <iostream>
#include <string>
#include <cstring>    // for memcpy
#include <vector>

#include <glog/logging.h>

#include <numa.h>
#include <numaif.h>

#include <tbb/tick_count.h>

#include <cuda.h>
#include <cuda_runtime.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::memcpy;
size_t count = 1024 * 1024;
size_t size = count * sizeof(float);
size_t rounds = 100000;
void *cpu_ptr;
void *cpu_ptr2;
void *gpu_ptr;
void *gpu_ptr2;

inline void *set_mem_affinity(int node_id) {
  bitmask *old_mask = numa_get_membind();
  CHECK(old_mask);
  bitmask *mask = numa_allocate_nodemask();
  CHECK(mask);
  mask = numa_bitmask_setbit(mask, node_id);
  numa_set_bind_policy(1); /* set NUMA zone binding to be "strict" */
  numa_set_membind(mask);
  numa_free_nodemask(mask);
  return reinterpret_cast<void *>(old_mask);
}

inline void restore_mem_affinity(void *mask_opaque) {
  bitmask *mask = reinterpret_cast<bitmask *>(mask_opaque);
  CHECK(mask);
  numa_set_bind_policy(0); /* set NUMA zone binding to be "preferred" */
  numa_set_membind(mask);
}

static void *thread_run(void *arg) {
  size_t thread_id = static_cast<size_t>((unsigned long)(arg));
  // void *local_cpu_ptr = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr)[start]);
  // void *local_cpu_ptr2 = reinterpret_cast<void *>(&reinterpret_cast<char *>(cpu_ptr2)[start]);
  // for (size_t r = 0; r < rounds; r++) {
    // memcpy(local_cpu_ptr2, local_cpu_ptr, local_size);
  // }
  cudaStream_t stream;
  cudaError_t result;
  result = cudaStreamCreate(&stream);
  tbb::tick_count tick_start = tbb::tick_count::now();
  double last_time = 0.0;
  for (size_t r = 0; r < rounds; r++) {
    if (thread_id == 0) {
      cudaMemcpyAsync(gpu_ptr, cpu_ptr, size, cudaMemcpyDefault, stream);
      cudaStreamSynchronize(stream);
      if ((r + 1) % 1024 == 0) {
        double time = (tbb::tick_count::now() - tick_start).seconds();
        double bandwidth = 4 / (time - last_time);
        double ave_bandwidth = 4 * (r + 1) / 1024 / time;
        cout << 4 * r << " MB memory CPU->GPU copied in " << time << ", bandwidth = " << bandwidth << ", ave_bandwidth = " << ave_bandwidth << endl;
        last_time = time;
      }
    } else {
      cudaMemcpyAsync(cpu_ptr2, gpu_ptr2, size, cudaMemcpyDefault, stream);
      cudaStreamSynchronize(stream);
      if ((r + 1) % 1024 == 0) {
        cout << 4 * r << " MB memory GPU->CPU copied in "
            << (tbb::tick_count::now() - tick_start).seconds() << endl;
      }
    }
  }
  result = cudaStreamDestroy(stream);
}

int main(int argc, char* argv[]) {
  int numa_node_id = atoi(argv[1]);
  void *opaque;
  if (numa_node_id >= 0) {
    opaque = set_mem_affinity(numa_node_id);
    restore_mem_affinity(opaque);
    opaque = set_mem_affinity(numa_node_id);
    restore_mem_affinity(opaque);
    opaque = set_mem_affinity(numa_node_id);
    restore_mem_affinity(opaque);
    opaque = set_mem_affinity(numa_node_id);
    restore_mem_affinity(opaque);
    opaque = set_mem_affinity(numa_node_id);
  }
  cudaMallocHost(&cpu_ptr, size);
  cudaMallocHost(&cpu_ptr2, size);
  cudaMalloc(&gpu_ptr, size);
  cudaMalloc(&gpu_ptr2, size);
  if (numa_node_id >= 0) {
    restore_mem_affinity(opaque);
  }

  thread_run(static_cast<size_t>(0));
}
