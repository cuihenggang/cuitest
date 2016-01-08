/*
    Copyright (c) 2007-2013 Contributors as noted in the AUTHORS file

    This file is part of 0MQ.

    0MQ is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    0MQ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <zmq.h>
#include <zmq_utils.h>

#include <mpi.h>

using namespace std;

struct config_t {
  void *ctx;
  int no_of_messages;
  int message_size;
};

void pusher_and_puller(int no_of_messages, int message_size) {
  void *ctx = zmq_init(1);
  assert(ctx);
  int rc;

  // Connect first
  void *push_socket = zmq_socket(ctx, ZMQ_PUSH);
  void *pull_socket = zmq_socket(ctx, ZMQ_PULL);
  assert(push_socket);
  assert(pull_socket);
  rc = zmq_connect(push_socket, "tcp://h0-dfge:9090");
  assert(rc == 0);
  rc = zmq_connect(pull_socket, "tcp://h0-dfge:9091");
  assert(rc == 0);

  // Create some data
  void *message = malloc(message_size);

  MPI_Barrier(MPI_COMM_WORLD);
  void *stopwatch = zmq_stopwatch_start();
  zmq_msg_t msg;
  rc = zmq_msg_init_size(&msg, message_size);
  memcpy(zmq_msg_data(&msg), message, message_size);
  for (int i = 0; i < no_of_messages; i ++) {
    rc = zmq_send(push_socket, &msg, 0);
    rc = zmq_msg_init(&msg);
    rc = zmq_recv(pull_socket, &msg, 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("pusher_and_puller time elapsed = %f\n", (double)elapsed / 1e6);

  free(message);

  // Cleanup
  rc = zmq_close(push_socket);
  rc = zmq_close(pull_socket);
  assert(rc == 0);
}

void puller_and_pusher(int no_of_messages, int message_size) {
  void *ctx = zmq_init(1);
  assert(ctx);
  int rc;

  // Bind first
  void *pull_socket = zmq_socket(ctx, ZMQ_PULL);
  void *push_socket = zmq_socket(ctx, ZMQ_PUSH);
  assert(pull_socket);
  assert(push_socket);
  rc = zmq_bind(pull_socket, "tcp://*:9090");
  assert(rc == 0);
  rc = zmq_bind(push_socket, "tcp://*:9091");
  assert(rc == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  void *stopwatch = zmq_stopwatch_start();
  zmq_msg_t msg;
  for (unsigned int i = 0; i < no_of_messages; ++i) {
    rc = zmq_msg_init(&msg);
    rc = zmq_recv(pull_socket, &msg, 0);
    rc = zmq_send(push_socket, &msg, 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("puller_and_pusher time elapsed = %f\n", (double)elapsed / 1e6);

  // Cleanup
  rc = zmq_close(push_socket);
  rc = zmq_close(pull_socket);
  assert(rc == 0);
}

void test_latency(int proc_id, int no_of_messages, int message_size) {
  if (proc_id == 0) {
    puller_and_pusher(no_of_messages, message_size);
  } else {
    pusher_and_puller(no_of_messages, message_size);
  }
}

int main (int argc, char *argv[]) {
  int proc_id;
  int numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  cout << "proc " << proc_id << endl;

  int test_type = argc > 1 ? atoi(argv[1]) : 0;
  int no_of_messages = argc > 2 ? atoi(argv[2]) : 10;
  int message_size = argc > 3 ? atoi(argv[3]) : 1 << 29;

  test_latency(proc_id, no_of_messages, message_size);

  return 0;
}
