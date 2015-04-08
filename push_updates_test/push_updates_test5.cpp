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

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <boost/thread.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_unordered_map.h>

#include <tbb/tick_count.h>

#include <zmq.hpp>

#include "zmq-portable-bytes.hpp"

using std::vector;
using std::string;
using std::cout;
using std::endl;

// #define VECTOR
// #define PAIR
// #define CONCURRENT
// #define OPLOG
// #define LOCK

int num_cols = 3;
int num_rows = 40000000;
int rounds = 1;

bool is_send_update = false;
bool is_disassemble = false;
bool is_pack_data = false;
bool is_send_msgs = false;

bool is_set_affinity = false;

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

// typedef VectorData RowData;
// typedef VectorData RowOpVal;

typedef val_t RowData;
typedef val_t RowOpVal;

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
    row_cvar(new boost::condition_variable()) {}
  SharedCacheRow& operator=(const SharedCacheRow&) {
    return *this;
  }
};
#if defined(VECTOR)
typedef std::vector<SharedCacheRow> SharedCache;
#elif defined(CONCURRENT)
  #if defined(PAIR)
typedef tbb::concurrent_unordered_map<TableRow, SharedCacheRow> SharedCache;
  #else
typedef tbb::concurrent_unordered_map<int, SharedCacheRow> SharedCache;
  #endif
#else
  #if defined(PAIR)
typedef boost::unordered_map<TableRow, SharedCacheRow> SharedCache;
  #else
typedef boost::unordered_map<int, SharedCacheRow> SharedCache;
  #endif
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

inline void pack_data(PortableBytes& bytes, const val_t& data) {
  bytes.pack<val_t>(data);
}

inline void unpack_data(val_t& data, PortableBytes& bytes) {
  bytes.unpack<val_t>(data);
}

inline bool is_able_to_pack_const_data(const val_t& data) {
  return false;
}

inline void pack_const_data(PortableBytes& bytes, val_t& data) {
  // not allowed
  assert(0);
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
    if (is_able_to_pack_const_data(data)) {
      pack_const_data(msgs[1], data);
    } else {
      pack_data(msgs[1], data);
    }
    if (is_send_msgs) {
      send_msgs(dst, msgs);
    }
  }
}

//  converts cpu set mask to a bitmap mask representation
static uint64_t cpubitmask(cpu_set_t *csetmaskp)
{
  int i;
  uint64_t bmask;

  for (i=0, bmask=0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, csetmaskp))
      bmask |= (1<<i);
  }

  return bmask;
}

/* 
 * IN: pid, cpunum
 * OUT: 0: success, -1: error (errno set)
 */
int migrateone(int cpunum)
{
  int size = sizeof(cpu_set_t);
  cpu_set_t oldmask, newmask;
  uint64_t oldbmask, newbmask;
  int rc = 0;

  pid_t tid = gettid();
  rc = sched_getaffinity(tid, size, &oldmask);
  assert(rc == 0);

  oldbmask = cpubitmask(&oldmask);

  CPU_ZERO(&newmask);
  CPU_SET(cpunum, &newmask);
  newbmask = cpubitmask(&newmask);

  if (oldbmask != newbmask) {
    rc = sched_setaffinity(tid, size, &newmask);
  }
  assert(rc == 0);

  return rc;
} /* migrateone */

static void *test_run(void *arg) {
  unsigned long thread_id = (unsigned long)arg;
  if (is_set_affinity) {
    migrateone(thread_id);
  }

  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  SharedCache shared_cache;

  tick_start = tbb::tick_count::now();
#if defined(VECTOR)
  shared_cache.resize(num_rows);
#else
  shared_cache.rehash(num_rows * 2);
#endif
  for (int i = 0; i < num_rows; i ++) {
#if defined(PAIR)
    SharedCacheRow& shared_cache_row = shared_cache[TableRow(0, i)];
#else
    SharedCacheRow& shared_cache_row = shared_cache[i];
#endif
#if defined(OPLOG)
    RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
    RowData& data = shared_cache_row.data;
#endif
    // data.resize(num_cols);
    // for (int j = 0; j < num_cols; j ++) {
      // data[j] = j;
    // }
    data = 1;
  }
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  std::cout << "Takes " << total_time << " to init" << std::endl;
  
  // int bucket_count = shared_cache.bucket_count();
  // int max_size = 0;
  // for (int i = 0; i < bucket_count; i++) {
    // int bucket_size = shared_cache.bucket_size(i);
    // max_size = bucket_size > max_size ? bucket_size : max_size;
  // }
  // std::cout << "Bucket count: " << bucket_count << std::endl;
  // std::cout << "Max bucket size: " << max_size << std::endl;

  tick_start = tbb::tick_count::now();
  for (int r = 0; r < rounds; r ++) {
#if defined(VECTOR)
    for (int i = 0; i < shared_cache.size(); i++) {
      table_id_t table = 0;
      row_idx_t row = i;
      SharedCacheRow& shared_cache_row = shared_cache[i];
#else
    for (SharedCache::iterator table_row_it = shared_cache.begin();
         table_row_it != shared_cache.end(); table_row_it++) {
  #if defined(PAIR)
      table_id_t table = table_row_it->first.first;
      row_idx_t row = table_row_it->first.second;
  #else
      table_id_t table = 0;
      row_idx_t row = table_row_it->first;
  #endif
      SharedCacheRow& shared_cache_row = table_row_it->second;
#endif
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
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  std::cout << "Takes " << total_time << " to send" << std::endl;
}

int main(int argc, char* argv[])
{
  int nr_threads = 1;

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
  if (argc > 2) {
    nr_threads = atoi(argv[2]);
    num_rows /= nr_threads;
  }
  if (argc > 3) {
    is_set_affinity = atoi(argv[3]) > 0;
  }

  vector<pthread_t> threads_id(nr_threads);
  vector<pthread_attr_t> threads_attr(nr_threads);
  void *res;
  for (int i = 0; i < nr_threads; i++) {
    pthread_attr_init(&threads_attr[i]);
    pthread_create(&threads_id[i], &threads_attr[i], test_run, (void *)i);
  }

  for (int i = 0; i < nr_threads; i++) {
    pthread_join(threads_id[i], &res);
  }
}