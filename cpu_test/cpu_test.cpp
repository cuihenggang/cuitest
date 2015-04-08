#include <assert.h>
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
// #include <numa.h>
// #include <numaif.h>

#include <iostream>
#include <vector>

#include <tbb/tick_count.h>

using std::vector;
using std::cout;
using std::endl;

int rounds = 1000;
int num_threads = 1;
int num_cores = 64;

double init_time = 0;
double send_time = 0;

bool is_set_cpu_affinity = false;
bool is_set_numa_mem_bind = false;

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
  // struct bitmask *mask = numa_allocate_nodemask();

  // assert(num_cores % num_threads == 0);
  // int div = num_cores / num_threads;
  // for (int i = thread_id * div; i < (thread_id + 1) * div; i++) {
    // mask = numa_bitmask_setbit(mask, i / 8);
  // }
	// numa_set_bind_policy(1); // set NUMA zone binding to be strict
  // numa_set_membind(mask);

  // numa_free_nodemask(mask);
  // return 0;
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

  int val = 8593759;
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    val = val * val % 10000000;
  }
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  std::cout << thread_id << " takes " << total_time << " to send" << std::endl;
  send_time += total_time;
  
  return (void *)val;
}

int main(int argc, char* argv[])
{
  if (argc > 1) {
    rounds = atoi(argv[1]);
    // num_rows /= num_threads;
  }
  if (argc > 2) {
    num_threads = atoi(argv[2]);
    // num_rows /= num_threads;
  }
  if (argc > 3) {
    int mode = atoi(argv[3]);
    is_set_cpu_affinity = mode > 0;
    is_set_numa_mem_bind = mode > 1;
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
  std::cout << "On average, Takes "<< init_time / num_threads
            << " to init" << std::endl;
  std::cout << "On average, Takes "<< send_time / num_threads
            << " to send" << std::endl;
}