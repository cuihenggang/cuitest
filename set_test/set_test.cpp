#include <iostream>
#include <stdlib.h>
#include <set>
#include <boost/unordered_set.hpp>

#include <tbb/tick_count.h>

int main(int argc, char* argv[])
{
  int num_entries = 10;
  int rounds = 100000;
  if (argc == 3) {
    num_entries = atoi(argv[1]);
    rounds = atoi(argv[2]);
  }
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  std::set<int> set;
  boost::unordered_set<int> uset;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < num_entries; i ++) {
    set.insert(i);
  }
  for (int i = 0; i < rounds; i ++) {
    set.erase(i);
    set.insert(i + num_entries);
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Std set takes " << total_time << " seconds" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < num_entries; i ++) {
    uset.insert(i);
  }
  for (int i = 0; i < rounds; i ++) {
    uset.erase(i);
    uset.insert(i + num_entries);
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Boost unordered set takes " << total_time << " seconds" << std::endl;
}
