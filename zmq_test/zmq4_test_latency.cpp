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

#include "testutil.hpp"

#define INPROC  0
#define TCP     1

struct config_t {
  void *ctx;
  int conn_type;
  int no_of_messages;
  int message_size;
  int no_of_threads;
};

static void pusher_and_puller(void *config) {
  void *ctx = ((config_t *)config)->ctx;
  int conn_type = ((config_t *)config)->conn_type;
  int no_of_messages = ((config_t *)config)->no_of_messages;
  int message_size = ((config_t *)config)->message_size;

  // Connect first
  void *push_socket = zmq_socket(ctx, ZMQ_PUSH);
  void *pull_socket = zmq_socket(ctx, ZMQ_PULL);
  assert(push_socket);
  assert(pull_socket);
  int rc = -1;
  if (conn_type == INPROC) {
    rc = zmq_connect(push_socket, "inproc://a");
    rc = zmq_connect(pull_socket, "inproc://b");
  } else if (conn_type == TCP) {
    rc = zmq_connect(push_socket, "tcp://127.0.0.1:9090");
    rc = zmq_connect(pull_socket, "tcp://127.0.0.1:9091");
  }
  assert(rc == 0);
  
  // Queue up some data
  void * message = malloc(message_size);
  
  void *stopwatch = zmq_stopwatch_start();
  for (int i = 0; i < no_of_messages; i ++) {
    rc = zmq_send (push_socket, message, message_size, 0);
    zmq_msg_t msg;
    rc = zmq_msg_init(&msg);
    rc = zmq_msg_recv(&msg, pull_socket, 0);
  }
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("sender time elapsed = %f\n", (double)elapsed / 1e6);
  
  free(message);
  
  // Cleanup
  rc = zmq_close(push_socket);
  rc = zmq_close(pull_socket);
  assert(rc == 0);
}

void test_latency(int conn_type,
                  int no_of_messages, int message_size,
                  unsigned int no_of_threads) {
  void *ctx = zmq_ctx_new ();
  assert(ctx);

  config_t config;
  config.ctx = ctx;
  config.conn_type = conn_type;
  config.no_of_messages = no_of_messages;
  config.message_size = message_size;
  config.no_of_threads = no_of_threads;

  int rc = -1;
  void **threads;
  threads = (void **)malloc(sizeof(void *) * no_of_threads);

  // Connect first
  for (unsigned int i = 0; i < no_of_threads; ++i) {
    threads[i] = zmq_threadstart(&pusher_and_puller, &config);
  }

  // Now bind
  void *pull_socket = zmq_socket(ctx, ZMQ_PULL);
  void *push_socket = zmq_socket(ctx, ZMQ_PUSH);
  assert(pull_socket);
  assert(push_socket);
  if (conn_type == INPROC) {
    rc = zmq_bind(pull_socket, "inproc://a");
    rc = zmq_bind(push_socket, "inproc://b");
  } else if (conn_type == TCP) {
    rc = zmq_bind(pull_socket, "tcp://*:9090");
    rc = zmq_bind(push_socket, "tcp://*:9091");
  }
  assert(rc == 0);
  
  void *stopwatch = zmq_stopwatch_start();
  
  for (unsigned int i = 0; i < no_of_threads * no_of_messages; ++i) {
    // Read pending message
    zmq_msg_t msg;
    rc = zmq_msg_init(&msg);
    rc = zmq_msg_recv(&msg, pull_socket, 0);
    rc = zmq_msg_send(&msg, push_socket, 0);
  }
  
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("receiver time elapsed = %f\n", (double)elapsed / 1e6);

  // Cleanup
  for (unsigned int i = 0; i < no_of_threads; ++i) {
    zmq_threadclose(threads [i]);
  }

  rc = zmq_close(pull_socket);
  rc = zmq_close(push_socket);
  assert(rc == 0);

  rc = zmq_ctx_term(ctx);
  assert(rc == 0);
  
  free(threads);
}

static void pusher (void *config) {
  void *ctx = ((config_t *)config)->ctx;
  int conn_type = ((config_t *)config)->conn_type;
  int no_of_messages = ((config_t *)config)->no_of_messages;
  int message_size = ((config_t *)config)->message_size;

  // Connect first
  void *connectSocket = zmq_socket(ctx, ZMQ_PUSH);
  assert(connectSocket);
  int rc = -1;
  if (conn_type == INPROC) {
    rc = zmq_connect(connectSocket, "inproc://a");
  } else if (conn_type == TCP) {
    rc = zmq_connect(connectSocket, "tcp://127.0.0.1:9090");
  }
  assert(rc == 0);
  
  // Queue up some data
  void * message = malloc(message_size);
  
  void *stopwatch = zmq_stopwatch_start();
  for (int i = 0; i < no_of_messages; i ++) {
    rc = zmq_send (connectSocket, message, message_size, 0);
  }
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("sender time elapsed = %f\n", (double)elapsed / 1e6);
  
  free(message);
  
  // Cleanup
  rc = zmq_close(connectSocket);
  assert(rc == 0);
}

void test_throughput(int conn_type,
                     int no_of_messages, int message_size,
                     unsigned int no_of_threads) {
  void *ctx = zmq_ctx_new ();
  assert(ctx);

  config_t config;
  config.ctx = ctx;
  config.conn_type = conn_type;
  config.no_of_messages = no_of_messages;
  config.message_size = message_size;
  config.no_of_threads = no_of_threads;

  int rc = -1;
  void **threads;
  threads = (void **)malloc(sizeof(void *) * no_of_threads);

  // Connect first
  for (unsigned int i = 0; i < no_of_threads; ++i) {
    threads[i] = zmq_threadstart(&pusher, &config);
  }

  // Now bind
  void *bindSocket = zmq_socket(ctx, ZMQ_PULL);
  assert(bindSocket);
  if (conn_type == INPROC) {
    rc = zmq_bind(bindSocket, "inproc://a");
  } else if (conn_type == TCP) {
    rc = zmq_bind(bindSocket, "tcp://*:9090");
  }
  assert(rc == 0);
  
  void *stopwatch = zmq_stopwatch_start();
  
  for (unsigned int i = 0; i < no_of_threads * no_of_messages; ++i) {
    // Read pending message
    zmq_msg_t msg;
    rc = zmq_msg_init(&msg);
    rc = zmq_msg_recv(&msg, bindSocket, 0);
  }
  
  unsigned long elapsed = zmq_stopwatch_stop(stopwatch);
  printf("receiver time elapsed = %f\n", (double)elapsed / 1e6);

  // Cleanup
  for (unsigned int i = 0; i < no_of_threads; ++i) {
    zmq_threadclose(threads [i]);
  }

  rc = zmq_close(bindSocket);
  assert(rc == 0);

  rc = zmq_ctx_term(ctx);
  assert(rc == 0);
  
  free(threads);
}

int main (int argc, char *argv[]) {
  printf("my test:\n");
  
  setup_test_environment();

  int test_type = argc > 1 ? atoi(argv[1]) : 0;
  int conn_type = argc > 2 ? atoi(argv[2]) : INPROC;
  int no_of_messages = argc > 3 ? atoi(argv[3]) : 10000;
  int message_size = argc > 4 ? atoi(argv[4]) : 400;
  int no_of_threads = argc > 5 ? atoi(argv[5]) : 1;
  
  if (test_type == 0) {
    test_throughput(conn_type, no_of_messages, message_size, no_of_threads);
  } else {
    test_latency(conn_type, no_of_messages, message_size, no_of_threads);
  }

  return 0;
}
