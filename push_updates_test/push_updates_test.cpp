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
#include <numa.h>
#include <numaif.h>

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <boost/thread.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/make_shared.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tick_count.h>

#include <zmq.hpp>

#include "zmq-portable-bytes.hpp"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using boost::make_shared;

// #define VECTOR_DATA
#define SHARED_CACHE
// #define VECTOR
// #define PAIR
// #define CONCURRENT
// #define OPLOG
// #define LOCK

int num_cols = 3;
int rounds = 100;
int num_rows = 40000000 / 64 / 100;
int num_threads = 1;
int num_cores = 64;

double init_time = 0;
double send_time = 0;

bool is_send_update = false;
bool is_disassemble = false;
bool is_pack_data = false;
bool is_send_msgs = false;

bool is_set_cpu_affinity = false;
bool is_set_numa_mem_bind = false;

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

#if defined(VECTOR_DATA)
typedef VectorData RowData;
typedef VectorData RowOpVal;
#else
typedef val_t RowData;
typedef val_t RowOpVal;
#endif

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
  boost::mutex row_mutex;
  boost::condition_variable row_cvar;
  SharedCacheRow() :
    data_age(INITIAL_DATA_AGE),
    tablet_server_id(-1) {}
  SharedCacheRow& operator=(const SharedCacheRow&) {
    return *this;
  }
};
typedef boost::shared_ptr<SharedCacheRow> SharedCacheRowPtr;
#if defined(VECTOR)
typedef std::vector<SharedCacheRowPtr> SharedCache;
#elif defined(CONCURRENT)
  #if defined(PAIR)
typedef tbb::concurrent_unordered_map<TableRow, SharedCacheRowPtr> SharedCache;
  #else
typedef tbb::concurrent_unordered_map<int, SharedCacheRowPtr> SharedCache;
  #endif
#else
  #if defined(PAIR)
typedef boost::unordered_map<TableRow, SharedCacheRowPtr> SharedCache;
  #else
typedef boost::unordered_map<int, SharedCacheRowPtr> SharedCache;
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

/* 
 * IN: pid, cpunum
 * OUT: 0: success, -1: error (errno set)
 */
int migrateone(int thread_id)
{
  int rc = 0;
  cpu_set_t mask;
  CPU_ZERO(&mask);

  assert(num_cores % num_cores == 0);
  int div = num_cores / num_threads;
  for (int i = thread_id * div; i < (thread_id + 1) * div; i++) {
    CPU_SET(i, &mask);
  }
	rc = sched_setaffinity(gettid(), sizeof(mask), &mask);
  assert(rc == 0);

  return 0;
} /* migrateone */

int numa_binds(int thread_id)
{
  struct bitmask *mask = numa_allocate_nodemask();

  assert(num_cores % num_threads == 0);
  int div = num_cores / num_threads;
  for (int i = thread_id * div; i < (thread_id + 1) * div; i++) {
    mask = numa_bitmask_setbit(mask, i / 8);
  }
	numa_set_bind_policy(1); // set NUMA zone binding to be strict
  numa_set_membind(mask);

  numa_free_nodemask(mask);
  return 0;
}

#if defined(SHARED_CACHE)
SharedCache shared_cache;
#endif

void init(SharedCache& shared_cache) {
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;

  tick_start = tbb::tick_count::now();
#if defined(VECTOR)
  shared_cache.resize(num_rows);
#else
  // shared_cache.rehash(num_rows * 2);
#endif
  for (int i = 0; i < num_rows; i ++) {
#if defined(PAIR)
    SharedCacheRowPtr& shared_cache_row_ptr = shared_cache[TableRow(0, i)];
    shared_cache_row_ptr = make_shared<SharedCacheRow>();
    SharedCacheRow& shared_cache_row = *shared_cache_row_ptr;
#else
    SharedCacheRowPtr& shared_cache_row_ptr = shared_cache[i];
    shared_cache_row_ptr = make_shared<SharedCacheRow>();
    SharedCacheRow& shared_cache_row = *shared_cache_row_ptr;
#endif
#if defined(OPLOG)
    RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
    RowData& data = shared_cache_row.data;
#endif
#if defined(VECTOR_DATA)
    data.resize(num_cols);
    for (int j = 0; j < num_cols; j ++) {
      data[j] = j;
    }
#else
    data = 1;
#endif
  }
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  std::cout << "Takes " << total_time << " to init" << std::endl;
  init_time += total_time;
}

static void *test_run(void *arg) {
  unsigned long thread_id = (unsigned long)arg;
  if (is_set_cpu_affinity) {
    migrateone(thread_id);
  }
  if (is_set_numa_mem_bind) {
    numa_binds(thread_id);
  }

#if !defined(SHARED_CACHE)
  SharedCache shared_cache;
  init(shared_cache);
#endif

  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  double total_time = 0;
  
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
      SharedCacheRow& shared_cache_row = *(shared_cache[i]);
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
      SharedCacheRow& shared_cache_row = *(table_row_it->second);
#endif
#if defined(LOCK)
      boost::unique_lock<boost::mutex> rowlock(*(shared_cache_row.row_mutex));
#endif
#if defined(OPLOG)
      RowOpVal& data = shared_cache_row.oplog[0].opval;
#else
      RowData& data = shared_cache_row.data;
#endif
      data++;
      // if (is_send_update) {
      if (!data) {
        string dst("local");
        send_update(dst, table, row, 0, data);
      }
    }
  }
  total_time = (tbb::tick_count::now() - tick_start).seconds();
  std::cout << "Takes " << total_time << " to send" << std::endl;
  send_time += total_time;
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
  if (argc > 2) {
    num_threads = atoi(argv[2]);
    // num_rows /= num_threads;
  }
  if (argc > 3) {
    int mode = atoi(argv[3]);
    is_set_cpu_affinity = mode > 0;
    is_set_numa_mem_bind = mode > 1;
  }

#if defined(SHARED_CACHE)
  init(shared_cache);
#endif

  vector<pthread_t> threads_id(num_threads);
  pthread_attr_t thread_attr;
  void *res;
  for (int i = 0; i < num_threads; i++) {
    pthread_attr_init(&thread_attr);
    pthread_create(&threads_id[i], &thread_attr, test_run, (void *)i);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads_id[i], &res);
  }
  std::cout << "On average, Takes "<< init_time / num_threads
            << " to init" << std::endl;
  std::cout << "On average, Takes "<< send_time / num_threads
            << " to send" << std::endl;
}