#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>
// #include <boost/vector.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

// #define LOCK
#define OPLOG
// #define CONCURRENT

#define num_cols 3
#define num_rows 1000000
#define rounds 20


#define BIG_ITER         1000000000
#define INITIAL_DATA_AGE -BIG_ITER
#define BIG_STALENESS    2 * BIG_ITER

typedef uint64_t row_idx_t;
typedef uint64_t col_idx_t;
typedef double val_t;
typedef uint32_t table_id_t;
typedef int64_t iter_t;

typedef std::pair<table_id_t, row_idx_t> TableRow;
typedef std::vector<val_t> VectorData;

typedef VectorData RowData;
typedef VectorData RowOpVal;

struct RowOp {
  RowOpVal opval;
  enum {
    NONE,
    INC,
  } flag;       /* 0 for nop, 1 for inc, 2 for put */
  RowOp() : flag(NONE) {}
};
typedef boost::unordered_map<TableRow, RowOp> TablesOp;
typedef boost::unordered_map<iter_t, RowOp> RowOpLog;

struct SharedCacheRow {
  RowData data;
  iter_t data_age;
  iter_t self_clock;
  /* Updates up to self_clock should be already applied in the data */
  RowOpLog oplog;
  int tablet_server_id;
  double rsponse_time;
  std::set<iter_t> fetches;
      /* On the fly requested data ages, this also includes pending_fetches */
  std::set<iter_t> pending_fetches;
      /* Data ages that we should send requests for after
       * the FIND_ROW request is replied
       */
  boost::shared_ptr<boost::mutex> row_mutex;
  boost::shared_ptr<boost::condition_variable> row_cvar;
  SharedCacheRow() :
    data_age(INITIAL_DATA_AGE),
    tablet_server_id(-1),
    row_mutex(new boost::mutex()),
    row_cvar(new boost::condition_variable()) {
  }
};
// typedef tbb::concurrent_unordered_map<TableRow, SharedCacheRow> SharedCache;
#if defined(CONCURRENT)
typedef tbb::concurrent_unordered_map<int, SharedCacheRow> SharedCache;
#else
typedef boost::unordered_map<int, SharedCacheRow> SharedCache;
#endif

int main(int argc, char* argv[])
{
  // int num_cols = atoi(argv[1]);
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  SharedCache shared_cache;

  for (int i = 0; i < num_rows; i ++) {
    // TableRow table_row(0, i);
    // SharedCacheRow& shared_cache_row = shared_cache[table_row];
    SharedCacheRow& shared_cache_row = shared_cache[i];
#if defined(OPLOG)
    RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
    RowData& data = shared_cache_row.data;
#endif
    data.resize(num_cols);
    for (int j = 0; j < num_cols; j ++) {
      data[j] = j;
    }
  }
  
  tick_start = tbb::tick_count::now();
  
  for (int r = 0; r < rounds; r ++) {
    for (SharedCache::iterator table_row_it = shared_cache.begin();
         table_row_it != shared_cache.end(); table_row_it++) {
      SharedCacheRow& shared_cache_row = table_row_it->second;
#if defined(LOCK)
      boost::unique_lock<boost::mutex> rowlock(*(shared_cache_row.row_mutex));
#endif
#if defined(OPLOG)
      RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
      RowData& data = shared_cache_row.data;
#endif
      
      // for (int j = 0; j < num_cols; j ++) {
        // data[j] ++;
      // }
    }
  }

  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << std::endl;
}