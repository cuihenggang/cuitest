/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Lazy table shards server
#include "tablet-server.hpp"
#include "common/row-op-util.hpp"

#include "boost_serialization_unordered_map.hpp"
/* A hack of unordered_map serialization from Internet */

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;
using boost::format;
using boost::lexical_cast;
using boost::shared_ptr;
using boost::make_shared;

ClientServerDecode::ClientServerDecode(TabletStorage& storage_i,
                                       MetadataServer& metadata_server_i)
  : storage(storage_i), metadata_server(metadata_server_i) {
}

void ClientServerDecode::create_table(const string& src,
                                      const vector<string>& args) {
  if (args.size() != 2) {
    cerr << "Malformed C->S CREATE_TABLE:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 2);
  }

  if (args[0].size() != sizeof(cs_create_table_msg_t)) {
    cerr << "Malformed C->S CREATE_TABLE:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_create_table_msg_t));
  }
  cs_create_table_msg_t *cs_create_table_msg =
  reinterpret_cast<cs_create_table_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_create_table_msg->client_id;
  row_idx_t num_row = cs_create_table_msg->num_row;
  row_idx_t num_col = cs_create_table_msg->num_col;

  const string& name = args[1];

  storage.create_table(src, client_id, name, num_row, num_col);
}

void ClientServerDecode::find_row(const string& src,
                                  const vector<string>& args) {
  if (args.size() != 1) {
    cerr << "Malformed C->S FIND_ROW:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 1);
  }

  if (args[0].size() != sizeof(cs_find_row_msg_t)) {
    cerr << "Malformed C->S FIND_ROW:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_find_row_msg_t));
  }
  cs_find_row_msg_t *cs_find_row_msg =
    reinterpret_cast<cs_find_row_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_find_row_msg->client_id;
  table_id_t table = cs_find_row_msg->table;
  row_idx_t row = cs_find_row_msg->row;

  metadata_server.find_row(src, client_id, table, row);
}

void ClientServerDecode::inc_row(const string& src,
                                 const vector<string>& args) {
  if (args.size() < 2) {
    cerr << "Malformed C->S INC_ROW:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() >= 2);
  }

  if (args[0].size() != sizeof(cs_inc_msg_t)) {
    cerr << "Malformed C->S INC_ROW:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_inc_msg_t));
  }
  cs_inc_msg_t *cs_inc_msg =
    reinterpret_cast<cs_inc_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_inc_msg->client_id;
  table_id_t table = cs_inc_msg->table;
  row_idx_t row = cs_inc_msg->row;
  iter_t iter = cs_inc_msg->iter;

  RowData data;
  vector<val_t> vals;
  unpack_string_vec<val_t>(vals, args[1]);
  vector<col_idx_t> cols;
  unpack_string_vec<col_idx_t>(cols, args[2]);

  // put column indices and values back into one vector or map
  resize_maybe(data, cols.size());
  for (uint i = 0; i < cols.size(); i++) {
    data[cols[i]] = vals[i];
  }

  storage.inc_row(src, client_id, table, row, data, iter);
}

void ClientServerDecode::iterate(const string& src,
                                 const vector<string>& args) {
  if (args.size() != 1) {
    cerr << "Malformed C->S ITERATE:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 1);
  }

  if (args[0].size() != sizeof(cs_iterate_msg_t)) {
    cerr << "Malformed C->S ITERATE:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_iterate_msg_t));
  }
  cs_iterate_msg_t *cs_iterate_msg =
    reinterpret_cast<cs_iterate_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_iterate_msg->client_id;
  iter_t iter = cs_iterate_msg->iter;

  storage.iterate(src, client_id, iter);
}

void ClientServerDecode::read_row(const string& src,
                                  const vector<string>& args) {
  if (args.size() != 1) {
    cerr << "Malformed C->S READ_ROW:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 1);
    return;
  }

  if (args[0].size() != sizeof(cs_read_row_msg_t)) {
    cerr << "Malformed C->S READ_ROW:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_read_row_msg_t));
  }
  cs_read_row_msg_t *cs_read_row_msg =
    reinterpret_cast<cs_read_row_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_read_row_msg->client_id;
  table_id_t table = cs_read_row_msg->table;
  row_idx_t row = cs_read_row_msg->row;
  iter_t data_age = cs_read_row_msg->data_age;
  double request_time = cs_read_row_msg->request_time;

  storage.read_row(src, client_id, table, row, data_age, request_time);
}

void ClientServerDecode::add_access_info(const std::string& src,
                                         const std::vector<std::string>& args) {
  if (args.size() != 2) {
    cerr << "Malformed C->S ADD_ACCESS_INFO:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 2);
  }

  if (args[0].size() != sizeof(cs_add_access_info_msg_t)) {
    cerr << "Malformed C->S ADD_ACCESS_INFO:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_add_access_info_msg_t));
  }
  cs_add_access_info_msg_t *cs_add_access_info_msg =
        reinterpret_cast<cs_add_access_info_msg_t *>(
                                  const_cast<char *>(args[0].data())
                                  );
  uint client_id = cs_add_access_info_msg->client_id;

  vector<RowAccessInfo> access_info;
  unpack_string_vec<RowAccessInfo>(access_info, args[1]);

  metadata_server.add_access_info(src, client_id, access_info);
}

void ClientServerDecode::get_stats(const string& src,
                                   const vector<string>& args) {
  if (args.size() != 1) {
    cerr << "Malformed C->S GET_STATS:"
         << " args.size() = " << args.size()
         << endl;
    assert(args.size() == 1);
  }

  if (args[0].size() != sizeof(cs_get_stats_msg_t)) {
    cerr << "Malformed C->S GET_STATS:"
         << " args[0].size() = " << args[0].size()
         << endl;
    assert(args[0].size() == sizeof(cs_get_stats_msg_t));
  }
  cs_get_stats_msg_t *cs_get_stats_msg =
    reinterpret_cast<cs_get_stats_msg_t *>(const_cast<char *>(args[0].data()));
  uint client_id = cs_get_stats_msg->client_id;

  storage.get_stats(src, client_id, metadata_server);
}

void ClientServerDecode::decode_msg(const string& src,
                                    const vector<string>& msgs) {
  if (msgs.size() < 1) {
    cerr << "Received message has parts missing!" << endl;
    assert(msgs.size() >= 1);
  }

  command_t cmd = unpack_string<command_t>(msgs[0]);
  switch (cmd) {
  case CREATE_TABLE:
    create_table(src, msgs);
    break;
  case FIND_ROW:
    find_row(src, msgs);
    break;
  case INC_ROW:
    inc_row(src, msgs);
    break;
  case READ_ROW:
    read_row(src, msgs);
    break;
  case ITERATE:
    iterate(src, msgs);
    break;
  case ADD_ACCESS_INFO:
    add_access_info(src, msgs);
    break;
  case GET_STATS:
    get_stats(src, msgs);
    break;
  default:
    cerr << "Server received unknown command!" << endl;
    assert(0);
  }
}

void ClientServerDecode::router_callback(const string& src,
    const vector<string>& msgs) {
  decode_msg(src, msgs);
}

RouterHandler::RecvCallback ClientServerDecode::get_recv_callback() {
  return bind(&ClientServerDecode::router_callback, this, _1, _2);
}

void ServerClientEncode::iterate(const vector<string>& clients,
                                 uint32_t rank, iter_t iter) {
  vector<string> msgs;
  msgs.resize(1);

  msgs[0].resize(sizeof(sc_iterate_msg_t));
  sc_iterate_msg_t *sc_iterate_msg =
    reinterpret_cast<sc_iterate_msg_t *>(const_cast<char *>(msgs[0].data()));
  sc_iterate_msg->cmd = ITERATE;
  sc_iterate_msg->rank = rank;
  sc_iterate_msg->iter = iter;

  router_handler.direct_send_to(clients, msgs);
}

void ServerClientEncode::create_table(const string& client,
                                      const string& name,
                                      table_id_t id) {
  vector<string> msgs;
  msgs.resize(2);

  msgs[0].resize(sizeof(sc_create_table_msg_t));
  sc_create_table_msg_t* sc_create_table_msg =
  reinterpret_cast<sc_create_table_msg_t *>(const_cast<char *>(msgs[0].data()));
  sc_create_table_msg->cmd = CREATE_TABLE;
  sc_create_table_msg->id = id;

  msgs[1].assign(name);
  router_handler.direct_send_to(client, msgs);
}

void ServerClientEncode::find_row(const std::string& client,
                                  table_id_t table, row_idx_t row,
                                  uint32_t tablet_server_id) {
  vector<string> msgs;
  msgs.resize(1);

  msgs[0].resize(sizeof(sc_find_row_msg_t));
  sc_find_row_msg_t *sc_find_row_msg =
    reinterpret_cast<sc_find_row_msg_t *>(const_cast<char *>(msgs[0].data()));
  sc_find_row_msg->cmd = FIND_ROW;
  sc_find_row_msg->table = table;
  sc_find_row_msg->row = row;
  sc_find_row_msg->tablet_server_id = tablet_server_id;

  router_handler.direct_send_to(client, msgs);
}

void ServerClientEncode::read_row(const vector<string>& clients,
                                  table_id_t table, row_idx_t row,
                                  iter_t data_age,
                                  const vector<iter_t>& client_clocks,
                                  RowData& data,
                                  double request_time) {
  vector<string> msgs;
  msgs.resize(3);

  msgs[0].resize(sizeof(sc_read_row_msg_t));
  sc_read_row_msg_t *sc_read_row_msg =
    reinterpret_cast<sc_read_row_msg_t *>(const_cast<char *>(msgs[0].data()));
  sc_read_row_msg->cmd = READ_ROW;
  sc_read_row_msg->table = table;
  sc_read_row_msg->row = row;
  sc_read_row_msg->data_age = data_age;
  sc_read_row_msg->self_clock = 0;
  /* This field should be filled with the client-clock for each client */
  sc_read_row_msg->request_time = request_time;

  vector<col_idx_t> cols;
  vector<val_t> vals;
  disassemble(data, cols, vals);
  pack_string_vec<val_t>(msgs[1], vals);
  pack_string_vec<col_idx_t>(msgs[2], cols);

  assert(clients.size() == client_clocks.size());
  for (uint i = 0; i < clients.size(); i ++) {
    sc_read_row_msg->self_clock = client_clocks[i];
    router_handler.direct_send_to(clients[i], msgs);
  }
}

void ServerClientEncode::read_row(const string& client,
                                  table_id_t table, row_idx_t row,
                                  iter_t data_age, iter_t client_clock,
                                  RowData& data, double request_time) {
  vector<string> clients;
  vector<iter_t> client_clocks;
  clients.push_back(client);
  client_clocks.push_back(client_clock);
  read_row(clients, table, row, data_age, client_clocks, data, request_time);
}

void ServerClientEncode::get_stats(const string& client,
                                   const string& stats) {
  vector<string> msgs;
  msgs.resize(2);

  msgs[0].resize(sizeof(sc_get_stats_msg_t));
  sc_get_stats_msg_t *sc_get_stats_msg =
    reinterpret_cast<sc_get_stats_msg_t *>(const_cast<char *>(msgs[0].data()));
  sc_get_stats_msg->cmd = GET_STATS;

  msgs[1].assign(stats);

  router_handler.direct_send_to(client, msgs);
}

string ServerClientEncode::get_router_stats() {
  return router_handler.get_stats();
}

void TabletStorage::read_row(const string& client, uint client_id,
                             table_id_t table, row_idx_t row,
                             iter_t iter, double request_time) {
  TableRow table_row(table, row);

  server_stats.nr_request++;
  if (client_id == rank) {
    server_stats.nr_local_request++;
  }
  if (sync_mode != ASYNC) {
    if (iter < current_iteration) {
      server_stats.nr_send++;
      communicator.read_row(client, table, row,
                            current_iteration - 1, vec_clock[client] - 1,
                            tables[table_row], request_time);

      /* Data age is "current_iteration - 1" because "data_age = i" means
       * "this row contains updates through iteration i"
       */
    } else {
      /* Queue request
       * If set entry "client" already exists, new entry will not be inserted.
       */
      pending_reads[iter][table_row].insert(client);
      nr_pending_reads++;
    }
  } else {
    /* Immediate reply */
    server_stats.nr_send++;
    communicator.read_row(client, table, row, current_iteration - 1,
                          vec_clock[client] - 1, tables[table_row],
                          request_time);
  }
}

void TabletStorage::create_table(const string& client, uint client_id,
                                 const string& table,
                                 row_idx_t num_row, col_idx_t num_col) {
  table_id_t tid;
  if (table_directory.count(table) > 0) {
    tid = table_directory[table];
  } else {
    tid = table_directory.size();
    table_directory[table] = tid;
    row_count[tid] = num_row;
    col_count[tid] = num_col;
    assert(num_col);
  }
  if (rank == 0) {
    /* Only table-server-0 sends this message */
    communicator.create_table(client, table, tid);
  }
}

void TabletStorage::inc_row(const string& client, uint client_id,
                            table_id_t table, row_idx_t row,
                            const RowData& row_data, iter_t iter_i) {
  server_stats.nr_update++;
  if (client_id == rank) {
    server_stats.nr_local_update++;
  }

  iter_t iter = vec_clock[client];
  if (iter_i != iter) {
    cerr << "WARNING CS clocks out of sync,"
         << " client = " << client
         << " iter_i = " << iter_i
         << " iter = " << iter
         << endl;
    assert(0);
  }

  RowData& op_row = pending_ops[client][iter].inc_log[make_pair(table, row)];
  op_row = row_data;
}

void TabletStorage::apply_ops(const string& client, iter_t iter) {
  Tables& inc_log = pending_ops[client][iter].inc_log;
  for (Tables::iterator table_row_it = inc_log.begin();
       table_row_it != inc_log.end(); table_row_it++) {
    RowData& row_data = tables[table_row_it->first];
    resize_maybe(row_data, col_count[table_row_it->first.first]);
    row_data += table_row_it->second;
  }

  pending_ops[client].erase(iter);
}

iter_t TabletStorage::clock_min() {
  iter_t cm = 0;
  bool init = false;
  typedef const pair<string, iter_t>& SI;
  BOOST_FOREACH(SI si, vec_clock) {
    if (!init || si.second < cm) {
      cm = si.second;
      init = true;
    }
  }
  return cm;
}

void TabletStorage::print_stats() {
  cerr << "Iteration " << current_iteration << ": ";
  typedef const pair<string, iter_t>& SI;
  BOOST_FOREACH(SI si, vec_clock) {
    cerr << si.second << " ";
  }
  cerr << endl;

  cerr << endl;
}

void TabletStorage::send_pending() {
  vector<iter_t> erase_i;
  for (ClientsPendingReads::iterator iter_it = pending_reads.begin();
       iter_it != pending_reads.end(); iter_it ++) {
    iter_t iter = iter_it->first;
    if (iter >= current_iteration) {
      continue;
    }
    if (iter != current_iteration - 1) {
      cerr << "WARNING: pending reads for old iteration: "
           << iter << " < " << current_iteration-1
           << endl;
    }
    erase_i.push_back(iter);

    for (ClientsPendingReadTables::iterator table_row_it =
           iter_it->second.begin();
         table_row_it != iter_it->second.end(); table_row_it ++) {
      table_id_t table = table_row_it->first.first;
      row_idx_t row = table_row_it->first.second;
      ClientSet& client_set = table_row_it->second;
      vector<string> clients;
      vector<iter_t> client_clocks;
      BOOST_FOREACH(const string& client, client_set) {
        clients.push_back(client);
        client_clocks.push_back(vec_clock[client] - 1);
      }
      /* All set, send data to all waiting clients */
      server_stats.nr_send += clients.size();
      communicator.read_row(clients, table, row,
                            current_iteration - 1 /* data age */,
                            client_clocks,
                            tables[make_pair(table, row)],
                            -1 /* it's a pending request */);
    }
  }
  /* Erase old requests */
  BOOST_FOREACH(iter_t i, erase_i) {
    pending_reads.erase(i);
  }
}

void TabletStorage::snapshot(double walltime) {
  std::string snapshot_path = log_output_dir;
  snapshot_path += "/snapshot.";
  snapshot_path +=
    boost::lexical_cast<std::string>(current_iteration - start_iter);
  snapshot_path += ".";
  snapshot_path += boost::lexical_cast<std::string>(rank);
  std::ofstream snapshot_out(snapshot_path.c_str());
  boost::archive::binary_oarchive oa(snapshot_out);
//  boost::archive::text_oarchive oa(snapshot_out);
  oa << tables;
  snapshot_out.close();

  std::string walltime_path = log_output_dir;
  walltime_path += "/walltime.";
  walltime_path += boost::lexical_cast<std::string>(rank);

  std::ofstream walltime_out(walltime_path.c_str(),
                             std::ofstream::out | std::ofstream::app);
  walltime_out << current_iteration - start_iter
               << "\t" << (walltime - start_time)
               << std::endl;
  walltime_out.close();

  if (rank == 0) {
    double end_time = MPI_Wtime();
    cerr << "iteration " << current_iteration
         << ", snapshot time = " << end_time - walltime << endl;
  }
}

void TabletStorage::log_iter(const string& client, iter_t iter_i) {
  double walltime = MPI_Wtime();
  if (client_iter_start_time == 0) {
    client_iter_start_time = walltime;
  }

  clockid_t cid;
  int s = pthread_getcpuclockid(pthread_self(), &cid);
  struct timespec ts;
  if (s == 0) {
    clock_gettime(cid, &ts);
  }
  char buffer[10];
  snprintf(buffer, sizeof(buffer),
           "%4ld.%03ld\n", ts.tv_sec, ts.tv_nsec / 1000000);

  std::string output_path = log_output_dir;
  output_path += "/process-iter.";
  output_path += boost::lexical_cast<std::string>(rank);
  std::ofstream output_stream(output_path.c_str(),
                              std::ofstream::out | std::ofstream::app);
  output_stream << "process_nr:" << client_id_map[client]
                << " walltime:" << (walltime - client_iter_start_time)
                << " iter:" << iter_i - 1
                << " shared_iter: " << current_iteration - 1
                << " nr_request: " << server_stats.nr_request
                << " nr_send: " << server_stats.nr_send
                << " nr_update: " << server_stats.nr_update
                << " cpu_time: " << buffer
                << std::endl;
  output_stream.close();
}

void TabletStorage::iterate(const string& client, uint client_id,
                            iter_t iter_i) {
  int timing = true;
  tbb::tick_count clock_ad_start;
  tbb::tick_count clock_ad_apply_op_end;
  tbb::tick_count clock_ad_end;

  if (timing) {
    clock_ad_start = tbb::tick_count::now();
  }

  if (!client_id_map.count(client)) {
    client_id_map[client] = client_id;
  }

  if (log_interval && iter_i > start_iter
      && (iter_i - start_iter) % log_interval == 0) {
    log_iter(client, iter_i);
  }

  vec_clock[client]++;
  if (vec_clock[client] != iter_i) {
    cerr << "clock " << client << " = " << vec_clock[client]
         << " should be " << iter_i
         << endl;
  }
  assert(vec_clock[client] == iter_i);

  /* Apply logs */
  apply_ops(client, iter_i - 1);

  iter_t iter = clock_min();
  if (iter < current_iteration) {
    cerr << "WARNING: iteration moved backwards: " << current_iteration
         << " -> " << iter << endl;
  } else if (current_iteration - iter > 1) {
    cerr << "WARNING: iteration jumped forward by " << current_iteration - iter
         << endl;
  }

  if (timing) {
    clock_ad_apply_op_end = tbb::tick_count::now();
    server_stats.clock_ad_apply_op_time +=
        (clock_ad_apply_op_end - clock_ad_start).seconds();
  }

  if (iter != current_iteration) {
    current_iteration = iter;
    if (rank == 0) {
      cerr << "Reached new iteration " << current_iteration << endl;
    }

    /* Send pending read requests */
    send_pending();

    /* Notify clients of new iteration */
    vector<string> clients;
    typedef pair<string, iter_t> SI;
    BOOST_FOREACH(SI si, vec_clock) {
      clients.push_back(si.first);
    }
    communicator.iterate(clients, rank, current_iteration);

    /* Flush table data to the disk */
    if (snapshot_interval && current_iteration > start_iter
        && (current_iteration - start_iter) % snapshot_interval == 0) {
      snapshot(MPI_Wtime());
    }

    if (current_iteration == start_iter) {
      start_time = MPI_Wtime();
    }
  }

  if (timing) {
    clock_ad_end = tbb::tick_count::now();
    server_stats.clock_ad_send_pending_time +=
      (clock_ad_end - clock_ad_apply_op_end).seconds();
    server_stats.clock_ad_time_tot +=
      (clock_ad_end - clock_ad_start).seconds();
  }
}

void TabletStorage::get_stats(const std::string& client, uint client_id,
                              MetadataServer& metadata_server) {
  /* Count # rows holding */
  uint count = 0;
  for (Tables::iterator table_row_it = tables.begin();
       table_row_it != tables.end(); table_row_it++) {
    count++;
  }
  server_stats.nr_rows = count;

  std::stringstream combined_server_stats;
  combined_server_stats << "{"
         << "\"storage\": " << server_stats.to_json() << ", "
         << "\"metadata\": " << metadata_server.get_stats() << ", "
         << "\"router\": " << communicator.get_router_stats()
         << " } ";
  communicator.get_stats(client, combined_server_stats.str());
}

void MetadataServer::add_access_info(const string& client, uint client_id,
                                     const AccessInfo& access_info) {
  if(policy == 3) {
    // cerr << "server " << rank
         // << " received access info from client " << client_id
         // << " with size " << access_info.size()
         // << endl;
    for (AccessInfo::const_iterator it = access_info.begin();
         it != access_info.end(); it++) {
      TableRow key(it->tid, it->rid);
      TargetServer server(client_id, it->nr_read, it->nr_write);
      // key not exist
      if (!row_tablet_map.count(key)) {
        row_tablet_map[key] = server;
        tablet_load[server.tid]++;
      }
      else {
        TargetServer& pre_server = row_tablet_map[key];
        // choose tablet with higher access frequency
        if (pre_server.nr_read + pre_server.nr_write 
            < server.nr_read + server.nr_write) {
          row_tablet_map[key] = server;
          // update load
          tablet_load[server.tid]++;
          tablet_load[pre_server.tid]--; 
        }
        // choose tablet with less load
        else if (pre_server.nr_read + pre_server.nr_write 
            == server.nr_read + server.nr_write
            && tablet_load[server.tid] < tablet_load[pre_server.tid]) { 
          row_tablet_map[key] = server; 
          // update load
          tablet_load[server.tid]++;
          tablet_load[pre_server.tid]--; 
        }
      }
    }

    nr_access_info_received++;
    if (nr_access_info_received == nr_tablets) {
      /* Received access info from all clients, now the row-to-tablet mapping
       * should be stable, and we can service FIND_ROW requests
       */
      decide_data_assignment();
      ready_to_serve = true;
      serve_pending_requests();
    }
  }
}

void MetadataServer::decide_data_assignment() {
  /* This is not needed for the current policy */
}

void MetadataServer::serve_pending_requests() {
  cerr << "server " << rank
       << " has " << pending_requests.size()
       << " pending requests"
       << endl;
  for (uint i = 0; i < pending_requests.size(); i ++) {
    FindRowRequest& request = pending_requests[i];
    TableRow key(request.table, request.row);
    uint tablet_server_id = row_tablet_map[key].tid;
    communicator.find_row(request.client, request.table, request.row, 
                          tablet_server_id);
  }
}

void MetadataServer::find_row(const string& client, uint client_id,
                              table_id_t table, row_idx_t row) {
  server_stats.nr_request++;

  uint tablet_server_id = 0;
  TableRow key(table, row);

  switch(policy) {
    // 1 - tablet server <-- row_id % nr_tablets
    case 1:
      tablet_server_id = row % nr_tablets;
      communicator.find_row(client, table, row, tablet_server_id);
      break;
    // 2 - tablet server <-- first accessing client
    case 2:
      if (row_tablet_map.find(key) == row_tablet_map.end()) {
        row_tablet_map[key].tid = client_id;  
      }
      tablet_server_id = row_tablet_map[key].tid;
      communicator.find_row(client, table, row, tablet_server_id);
      break;
    // 3 - max(local access) + load balancing
    case 3:
      if (ready_to_serve) {
        tablet_server_id = row_tablet_map[key].tid;
        communicator.find_row(client, table, row, tablet_server_id);
      } else {
        /* row-to-tablet mapping is not ready, save to pending requests */
        pending_requests.push_back(FindRowRequest(client, table, row));
      }
      break;
    default:  // send every request to tablet 0...
      communicator.find_row(client, table, row, tablet_server_id);
  }
}

string MetadataServer::get_stats() {
  return server_stats.to_json();
}

void server(shared_ptr<zmq::context_t> zmq_ctx,
    uint server_id, uint nr_servers,
    const ServerConfig& config) {
  uint nr_tablets = nr_servers;
  uint tablet_rank = server_id;
  string request_url = "tcp://*:9090";

  cerr << "Tablet " << tablet_rank
       << " out of " << nr_tablets
       << " started"
       << endl;

  vector<string> connect_i;   /* Empty connect to */
  vector<string> bind_i;
  bind_i.push_back(request_url);
  string tablet_name = (format("tablet-%i") % tablet_rank).str();
  shared_ptr<RouterHandler> router_handler =
    make_shared<RouterHandler>(zmq_ctx, connect_i, bind_i, tablet_name);

  ServerClientEncode encoder(*router_handler);

  TabletStorage storage(encoder, tablet_rank, nr_tablets, config);
  MetadataServer metadata_server(encoder, tablet_rank, nr_tablets, config);

  ClientServerDecode decoder(storage, metadata_server);

  router_handler->do_handler(decoder.get_recv_callback());
}
