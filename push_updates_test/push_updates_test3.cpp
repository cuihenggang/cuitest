#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <set>
#include <map>
// #include <boost/vector.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

#include <zmq.hpp>

#include "zmq-portable-bytes.hpp"

using std::vector;
using std::string;

#define LOCK
#define OPLOG
// #define CONCURRENT

#define num_cols 3
#define num_rows 40000000
#define rounds 1


#define BIG_ITER         1000000000
#define INITIAL_DATA_AGE -BIG_ITER
#define BIG_STALENESS    2 * BIG_ITER

bool is_send_update = false;
bool is_disassemble = false;
bool is_pack_data = false;
bool is_send_msgs = false;

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
typedef tbb::concurrent_unordered_map<TableRow, SharedCacheRow> SharedCache;
#else
typedef boost::unordered_map<TableRow, SharedCacheRow> SharedCache;
#endif

enum Command {
  CREATE_TABLE,
  FIND_ROW,
  READ_ROW,
  INC_ROW,
  ITERATE,
  ADD_ACCESS_INFO,
  GET_STATS,
  SHUTDOWN
};
typedef uint8_t command_t;

struct cs_inc_msg_t {
  command_t cmd;
  uint32_t client_id;
  table_id_t table;
  row_idx_t row;
  iter_t iter;
};

void move_pb_to_zmq(zmq::message_t& zmq_msg, ZmqPortableBytes& pb) {
  zmq_msg_t *zmq_msg_ptr = reinterpret_cast<zmq_msg_t *>(&zmq_msg);
  zmq_msg_t *pb_msg_ptr = pb.get_msg_ptr();
  zmq_msg_move(zmq_msg_ptr, pb_msg_ptr);
}

int send_msgs(string dst, vector<ZmqPortableBytes>& parts) {
  int ret;
  if (!dst.compare("local")) {
    for (int i = 0; i < parts.size(); i++) {
      zmq::message_t zmq_msg;
      move_pb_to_zmq(zmq_msg, parts[i]);
      ret += zmq_msg.size();
    }
  }
  return ret;
}

void pack_data(PortableBytes& bytes, const VectorData& data) {
  size_t size = data.size() * sizeof(val_t);
  bytes.init_size(size);
  memcpy(bytes.data(), data.data(), size);
}

void unpack_data(VectorData& data, PortableBytes& bytes) {
  size_t size = bytes.size() / sizeof(val_t);
  data.resize(size);
  memcpy(data.data(), bytes.data(), bytes.size());
}

bool is_able_to_pack_const_data(const VectorData& data) {
  return true;
}

void pack_const_data(PortableBytes& bytes, VectorData& data) {
  size_t size = data.size() * sizeof(data[0]);
  bytes.init_data(data.data(), size, NULL, NULL);
}

void send_update(string dst, table_id_t table, row_idx_t row, iter_t iter, RowOpVal& data) {
// void send_update(int row, iter_t iter, RowOpVal& data) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);
  msgs[0].init_size(sizeof(cs_inc_msg_t));
  cs_inc_msg_t *cs_inc_msg = reinterpret_cast<cs_inc_msg_t *>(msgs[0].data());
  cs_inc_msg->cmd = INC_ROW;
  cs_inc_msg->client_id = 0;
  cs_inc_msg->table = table;
  cs_inc_msg->row = row;
  cs_inc_msg->iter = iter;
  if (is_disassemble) {
    pack_const_data(msgs[1], data);
    if (is_send_msgs) {
      send_msgs(dst, msgs);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc > 1) {
    int mode = atoi(argv[1]);
    if (mode > 0) {
      is_send_update = true;
    }
    if (mode > 1) {
      is_disassemble = true;
    }
    if (mode > 2) {
      is_pack_data = true;
    }
    if (mode > 3) {
      is_send_msgs = true;
    }
  }
  
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
  SharedCache shared_cache;

  for (int i = 0; i < num_rows; i ++) {
    TableRow table_row(0, i);
    SharedCacheRow& shared_cache_row = shared_cache[table_row];
    // SharedCacheRow& shared_cache_row = shared_cache[i];
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
      table_id_t table = table_row_it->first.first;
      row_idx_t row = table_row_it->first.second;
      SharedCacheRow& shared_cache_row = table_row_it->second;
#if defined(LOCK)
      boost::unique_lock<boost::mutex> rowlock(*(shared_cache_row.row_mutex));
#endif
#if defined(OPLOG)
      RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
      RowData& data = shared_cache_row.data;
#endif

      if (is_send_update) {
        string dst("local");
        send_update(dst, table, row, 0, data);
      }
    }
  }

  tick_end = tbb::tick_count::now();
  total_time = (tick_end - tick_start).seconds();
  std::cout << "Concurrent map takes " << total_time << std::endl;
}