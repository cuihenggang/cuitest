#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

// #define count 100
// #define rounds 1000000

#define count 40000000
#define rounds 1

using std::pair;

int main(int argc, char* argv[])
{
  // int count = atoi(argv[1]);
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  typedef std::pair<int, int> pair_t;
  tbb::concurrent_unordered_map<int, int> cumap;
  boost::unordered_map<int, int> umap;
  std::map<int, int> map;
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
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    cumap[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    umap[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    map[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    pcumap[pair_t(0, i)] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Concurrent map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    pumap[pair_t(0, i)] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Unordered map takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    pmap[pair_t(0, i)] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Map takes " << total_time << " seconds to initialize" << std::endl;

  tick_start = tbb::tick_count::now();
  vec.resize(count);
  std::cout << "Vector tests zerofy: " << vec[count / 2] << std::endl;
  for (int i = 0; i < count; i ++) {
    vec[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Vector takes " << total_time << " seconds to initialize" << std::endl;

  tick_start = tbb::tick_count::now();
  array = new int[count];
  for (int i = 0; i < count; i ++) {
    array[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Array takes " << total_time << " seconds to initialize" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    static_array[i] = 0;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Static array takes " << total_time << " seconds to initialize" << std::endl;
  

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      cumap[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      umap[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      map[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      pcumap[pair_t(0, i)] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Concurrent map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      pumap[pair_t(0, i)] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Unordered map takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      pmap[pair_t(0, i)] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Pair Map takes " << total_time << " seconds to update" << std::endl;

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      vec[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Vector takes " << total_time << " seconds to update" << std::endl;

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      array[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Array takes " << total_time << " seconds to update" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < count; i ++) {
      static_array[i] ++;
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Static array takes " << total_time << " seconds to update" << std::endl;
  
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    cumap2 = cumap;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << " seconds to copy" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    umap2 = umap;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Unordered map takes " << total_time << " seconds to copy" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    map2 = map;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Map takes " << total_time << " seconds to copy" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
    vec2 = vec;
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Vector takes " << total_time << " seconds to copy" << std::endl;
  
  delete[] array;
}
