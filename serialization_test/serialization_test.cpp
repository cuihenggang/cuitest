#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include "boost_serialization_unordered_map.hpp"

typedef int64_t row_idx_t;
typedef int64_t col_idx_t;
typedef double val_t;
typedef int32_t table_id_t;

typedef std::map<col_idx_t, val_t> Row;
typedef std::map<row_idx_t, Row> Table;
typedef std::map<table_id_t, Table> Tables;

typedef boost::unordered_map<col_idx_t, val_t> URow;
typedef boost::unordered_map<row_idx_t, URow> UTable;
typedef boost::unordered_map<table_id_t, UTable> UTables;

int main(int argc, char* argv[])
{
  UTables tables_out;
  tables_out[0][1][2] = 1;
  std::string snapshot_path = "./snapshot.test";
  std::ofstream snapshot_out(snapshot_path.c_str(), std::ios::binary);
  boost::archive::binary_oarchive oa(snapshot_out);
  oa << tables_out;
  snapshot_out.close();

  UTables tables_in;
  std::ifstream snapshot_in(snapshot_path.c_str(), std::ios::binary);
  boost::archive::binary_iarchive ia(snapshot_in);
  ia >> tables_in;
  std::cout << "tables_in[0][1][2] = "
            << tables_in[0][1][2] << std::endl;
  snapshot_in.close();
}
