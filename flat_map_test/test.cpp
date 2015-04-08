#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/syscall.h>
#define gettid() syscall(__NR_gettid)
#include <numa.h>
#include <numaif.h>

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <boost/thread.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/make_shared.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tick_count.h>

#define VECTOR      0
#define UMAP        1
#define UMAP_ITER   2

using std::vector;
using std::string;
using std::cout;
using std::endl;
using boost::make_shared;

int num_cols = 3;
int rounds = 10;
int num_rows = 40000000;
int num_threads = 1;
int num_cores = 64;

double iter_time = 0;

int mode;

bool is_set_cpu_affinity = false;
bool is_set_numa_mem_bind = false;

vector<int> vec_cache;
boost::unordered_map<int, int> umap_cache;

/* 
 * IN: pid, cpunum
 * OUT: 0: success, -1: error (errno set)
 */
int migrateone(int thread_id)
{
  int rc = 0;
  cpu_set_t mask;
  CPU_ZERO(&mask);

  assert(num_cores % num_cores == 0);
  int div = num_cores / num_threads;
  for (int i = thread_id * div; i < (thread_id + 1) * div; i++) {
    CPU_SET(i, &mask);
  }
	rc = sched_setaffinity(gettid(), sizeof(mask), &mask);
  assert(rc == 0);

  return 0;
} /* migrateone */

int numa_binds(int thread_id)
{
  struct bitmask *mask = numa_allocate_nodemask();

  assert(num_cores % num_threads == 0);
  int div = num_cores / num_threads;
  for (int i = thread_id * div; i < (thread_id + 1) * div; i++) {
    mask = numa_bitmask_setbit(mask, i / 8);
  }
	numa_set_bind_policy(1); // set NUMA zone binding to be strict
  numa_set_membind(mask);

  numa_free_nodemask(mask);
  return 0;
}

static void *test_run(void *arg) {
  unsigned long thread_id = (unsigned long)arg;
  if (is_set_cpu_affinity) {
    migrateone(thread_id);
  }
  if (is_set_numa_mem_bind) {
    numa_binds(thread_id);
  }

  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r++) {
    if (mode == VECTOR) {
      for (int i = 0; i < num_rows; i++) {
        vec_cache[i]++;
      }
    } else if (mode == UMAP) {
      for (int i = 0; i < num_rows; i++) {
        umap_cache[i]++;
      }
    } else if (mode == UMAP_ITER) {
      for (boost::unordered_map<int, int>::iterator it = umap_cache.begin();
           it != umap_cache.end(); it++) {
        it->second++;
      }
    }
    // if (mode == VECTOR) {
      // for (int i = thread_id; i < num_rows; i += num_threads) {
        // vec_cache[i]++;
      // }
    // } else if (mode == UMAP) {
      // for (int i = thread_id; i < num_rows; i += num_threads) {
        // umap_cache[i]++;
      // }
    // }
  }
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  // std::cout << "Takes " << total_time << " to iter" << std::endl;
  iter_time += total_time;
}

int main(int argc, char* argv[])
{
  if (argc > 1) {
    mode = atoi(argv[1]);
  }
  if (argc > 2) {
    num_threads = atoi(argv[2]);
    num_rows /= num_threads;
  }
  if (argc > 3) {
    int mode = atoi(argv[3]);
    is_set_cpu_affinity = mode > 0;
    is_set_numa_mem_bind = mode > 1;
  }

  if (mode == VECTOR) {
    vec_cache.resize(num_rows);
  } else if (mode == UMAP || mode == UMAP_ITER) {
    for (int i = 0; i < num_rows; i++) {
      umap_cache[i] = 0;
    }
  }

  vector<pthread_t> threads_id(num_threads);
  pthread_attr_t thread_attr;
  void *res;
  for (int i = 0; i < num_threads; i++) {
    pthread_attr_init(&thread_attr);
    pthread_create(&threads_id[i], &thread_attr, test_run, (void *)i);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads_id[i], &res);
  }
  std::cout << "On average, Takes "<< iter_time / num_threads
            << " to iter" << std::endl;
}