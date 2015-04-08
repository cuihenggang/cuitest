#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>
// #include <boost/vector.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

#define num_cols 3
#define num_rows 1000000
#define rounds 20

int main(int argc, char* argv[])
{
  // int num_cols = atoi(argv[1]);
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  typedef std::vector<double> Row;
  typedef tbb::concurrent_unordered_map<int, Row> Table;
  // typedef boost::unordered_map<int, Row> Table;
  Table table;

  for (int i = 0; i < num_rows; i ++) {
    Row& row = table[i];
    row.resize(num_cols);
    for (int j = 0; j < num_cols; j ++) {
      row[j] = j;
    }
  }
  
  tick_start = tbb::tick_count::now();
  
  for (int r = 0; r < rounds; r ++) {
    for (Table::iterator it = table.begin();
         it != table.end(); it ++) {
      Row& row = it->second;
      for (int j = 0; j < num_cols; j ++) {
        row[j] ++;
      }
    }
  }

  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << std::endl;
  
  typedef std::vector<Row> Table2;
  Table2 table2;

  table2.resize(num_rows);
  for (int i = 0; i < num_rows; i ++) {
    Row& row = table2[i];
    row.resize(num_cols);
    for (int j = 0; j < num_cols; j ++) {
      row[j] = j;
    }
  }
  
  tick_start = tbb::tick_count::now();
  
  for (int r = 0; r < rounds; r ++) {
    for (int i = 0; i < num_rows; i ++) {
      Row& row = table2[i];
      for (int j = 0; j < num_cols; j ++) {
        row[j] ++;
      }
    }
  }

  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Vector takes " << total_time << std::endl;
}