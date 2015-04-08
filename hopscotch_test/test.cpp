#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>

#include <boost/thread.hpp>

#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

#include "hopscotch_map.hpp"

// #define count 100
// #define rounds 1000000

// #define count 40000000
#define count 2000000
#define rounds 1

using std::pair;

static void *test_run(void *arg) {
  // int count = atoi(argv[1]);
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  typedef std::pair<int, int> pair_t;
  graphlab::hopscotch_map<int, int> hmap;
  tbb::concurrent_unordered_map<int, int> cumap;
  boost::unordered_map<int, int> umap;
  std::map<int, int> map;
  graphlab::hopscotch_map<pair_t, int> phmap;
  tbb::concurrent_unordered_map<pair_t, int> pcumap;
  boost::unordered_map<pair_t, int> pumap;
  std::map<pair_t, int> pmap;
  std::vector<int> vec;
  tbb::concurrent_unordered_map<int, int> cumap2;
  boost::unordered_map<int, int> umap2;
  std::map<int, int> map2;
  std::vector<int> vec2;
  int *array;
  int static_array[count];
  
  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // hmap[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Hopscotch map takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // cumap[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Concurrent map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    umap[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to initialize" << std::endl;
  uint bucket_count = umap.bucket_count();
  uint max_size = 0;
  for (uint i = 0; i < bucket_count; i++) {
    uint bucket_size = umap.bucket_size(i);
    max_size = bucket_size > max_size ? bucket_size : max_size;
  }
  std::cout << "bucket_count = " << bucket_count << std::endl;
  std::cout << "max_bucket_size = " << max_size << std::endl;
  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // map[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Map takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // phmap[pair_t(0, i)] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Hopscotch map takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // pcumap[pair_t(0, i)] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Concurrent map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    pumap[pair_t(0, i)] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Unordered map takes " << total_time << " seconds to initialize" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // pmap[pair_t(0, i)] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Map takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // vec.resize(count);
  // std::cout << "Vector tests zerofy: " << vec[count / 2] << std::endl;
  // for (int i = 0; i < count; i ++) {
    // vec[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Vector takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // array = new int[count];
  // for (int i = 0; i < count; i ++) {
    // array[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Array takes " << total_time << " seconds to initialize" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int i = 0; i < count; i ++) {
    // static_array[i] = 0;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Static array takes " << total_time << " seconds to initialize" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // int& val = hmap[i];
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Hopscotch map takes " << total_time << " seconds to locate" << std::endl;

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      // int& val = umap[i];
      boost::unordered_map<int, int>::iterator it = umap.find(i);
      assert(it != umap.end());
      assert(it->second >= 0);
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to locate" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // int& val = phmap[pair_t(0, i)];
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Hopscotch map takes " << total_time << " seconds to locate" << std::endl;

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      int& val = pumap[pair_t(0, i)];
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Unordered map takes " << total_time << " seconds to locate" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // hmap[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Hopscotch map takes " << total_time << " seconds to update" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // cumap[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Concurrent map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      umap[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to update" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // map[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Map takes " << total_time << " seconds to update" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // phmap[pair_t(0, i)] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Hopscotch map takes " << total_time << " seconds to update" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // pcumap[pair_t(0, i)] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Concurrent map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      pumap[pair_t(0, i)] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Unordered map takes " << total_time << " seconds to update" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // pmap[pair_t(0, i)] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Pair Map takes " << total_time << " seconds to update" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // vec[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Vector takes " << total_time << " seconds to update" << std::endl;

  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // array[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Array takes " << total_time << " seconds to update" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // for (int i = 0; i < count; i ++) {
      // static_array[i] ++;
    // }
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Static array takes " << total_time << " seconds to update" << std::endl;
  
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // cumap2 = cumap;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Concurrent map takes " << total_time << " seconds to copy" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // umap2 = umap;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Unordered map takes " << total_time << " seconds to copy" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // map2 = map;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Map takes " << total_time << " seconds to copy" << std::endl;
  
  // tick_start = tbb::tick_count::now();
  // for (int r = 0; r < rounds; r ++) {
    // vec2 = vec;
  // }
  // tick_end = tbb::tick_count::now();
  // total_time = (tick_end - tick_start).seconds();
  // std::cout << "Vector takes " << total_time << " seconds to copy" << std::endl;
  
  delete[] array;
}

int main(int argc, char* argv[]) {
  int num_threads = 1;
  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }

  std::vector<pthread_t> threads_id(num_threads);
  pthread_attr_t thread_attr;
  void *res;
  for (int i = 0; i < num_threads; i++) {
    pthread_attr_init(&thread_attr);
    pthread_create(&threads_id[i], &thread_attr, test_run, (void *)i);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads_id[i], &res);
  }
}
