#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <tbb/tick_count.h>

using namespace std;

int main(int argc, char* argv[])
{
  int size = 400;
  int num_entries = 5;
  int rounds = 1000;
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  int *a = new int[size / 4];
  for (int i = 0; i < size / 4; i ++) {
    a[i] = i;
  }
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < rounds; i ++) {
    vector<string> sv;
    for (int i = 0; i < num_entries; i ++) {
      sv.push_back(string((char *)a, size));
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Approach 1 set takes " << total_time << " seconds" << std::endl;
  
  tick_start = tbb::tick_count::now();
  for (int i = 0; i < rounds; i ++) {
    vector<string> sv(num_entries);
    for (int i = 0; i < num_entries; i ++) {
      sv[i].assign((char *)a, size);
    }
  }
  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Approach 2 set takes " << total_time << " seconds" << std::endl;
}
