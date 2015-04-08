#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

#include <pthread.h>

#define count 1000000

using std::cout;
using std::endl;

boost::unordered_map<int, int> umap;

static void *thread_run(void *arg) {
  cout << "thread-2 starts" << endl;
  for (int i = 0; i < count; i ++) {
    umap[i]++;
  }
  cout << "thread-2 finishes" << endl;
}

int main(int argc, char* argv[])
{
  for (int i = 0; i < count; i ++) {
    umap[i] = 0;
  }

  pthread_t thread_id;
  pthread_attr_t thread_attr;
  void *res;
  pthread_attr_init(&thread_attr);
  pthread_create(&thread_id, &thread_attr, thread_run, NULL);

  cout << "thread-1 starts" << endl;
  for (int i = 0; i < count; i ++) {
    umap[i]++;
  }
  cout << "thread-1 finishes" << endl;

  pthread_join(thread_id, &res);
}
