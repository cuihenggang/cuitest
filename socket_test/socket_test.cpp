#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <time.h> 
#include <pthread.h>
#include <tbb/tick_count.h>

struct config_t {
  int no_of_messages;
  int message_size;
};

static void *client_latency(void *arg) {
  int no_of_messages = ((config_t *)arg)->no_of_messages;
  int message_size = ((config_t *)arg)->message_size;
  int sockfd = 0;
  struct sockaddr_in serv_addr; 

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Error : Could not create socket \n");
    return NULL;
  } 

  memset(&serv_addr, '0', sizeof(serv_addr)); 

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(5000); 

  void * message = malloc(message_size);
  
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    printf("\n inet_pton error occured\n");
    return NULL;
  }
  printf("connect\n");
  if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("\n Error : Connect Failed \n");
    return NULL;
  }

  tbb::tick_count tick_start = tbb::tick_count::now();
  for (int i = 0; i < no_of_messages; i ++) {
    write(sockfd, message, message_size);
    int bytes_received = 0;
    while (bytes_received < message_size) {
      bytes_received +=
        read(sockfd, message + bytes_received, message_size - bytes_received);
    }
  }
  tbb::tick_count tick_end = tbb::tick_count::now();
  double elapsed = (tick_end - tick_start).seconds();
  printf("sender time elapsed = %f\n", elapsed);

  close(sockfd);
  free(message);
  return NULL;
}

void test_latency(int no_of_messages, int message_size) {
  int listenfd = 0, connfd = 0;
  struct sockaddr_in serv_addr; 

  listenfd = socket(AF_INET, SOCK_STREAM, 0);
  memset(&serv_addr, '0', sizeof(serv_addr));

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_addr.sin_port = htons(5000); 

  bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 

  listen(listenfd, 10);
  
  /* Create client thread */
  pthread_t client_thread_id;
  pthread_attr_t client_attr;
  void *res;
  config_t config;
  config.no_of_messages = no_of_messages;
  config.message_size = message_size;
  pthread_attr_init(&client_attr);
  pthread_create(&client_thread_id, &client_attr, client_latency, &config);

  void *message = malloc(message_size);
  while (1) {
    connfd = accept(listenfd, (struct sockaddr*)NULL, NULL); 

    tbb::tick_count tick_start = tbb::tick_count::now();
    for (int i = 0; i < no_of_messages; i++) {
      int bytes_received = 0;
      while (bytes_received < message_size) {
        bytes_received +=
          read(connfd, message + bytes_received, message_size - bytes_received);
      }
      write(connfd, message, message_size);
    }
    tbb::tick_count tick_end = tbb::tick_count::now();
    double elapsed = (tick_end - tick_start).seconds();
    printf("receiver time elapsed = %f\n", elapsed);

    close(connfd);
    break;
  }
  
  close(listenfd);
  free(message);
  pthread_join(client_thread_id, &res);
}

static void *client(void *arg) {
  int no_of_messages = ((config_t *)arg)->no_of_messages;
  int message_size = ((config_t *)arg)->message_size;
  int sockfd = 0;
  struct sockaddr_in serv_addr; 

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Error : Could not create socket \n");
    return NULL;
  } 

  memset(&serv_addr, '0', sizeof(serv_addr)); 

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(5000); 

  void * message = malloc(message_size);
  
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    printf("\n inet_pton error occured\n");
    return NULL;
  }
  printf("connect\n");
  if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("\n Error : Connect Failed \n");
    return NULL;
  }

  tbb::tick_count tick_start = tbb::tick_count::now();
  for (int i = 0; i < no_of_messages; i ++) {
    write(sockfd, message, message_size);
  }
  tbb::tick_count tick_end = tbb::tick_count::now();
  double elapsed = (tick_end - tick_start).seconds();
  printf("sender time elapsed = %f\n", elapsed);

  close(sockfd);
  free(message);
  return NULL;
}

void test_throughput(int no_of_messages, int message_size) {
  int listenfd = 0, connfd = 0;
  struct sockaddr_in serv_addr; 

  listenfd = socket(AF_INET, SOCK_STREAM, 0);
  memset(&serv_addr, '0', sizeof(serv_addr));

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_addr.sin_port = htons(5000); 

  bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 

  listen(listenfd, 10);
  
  /* Create client thread */
  pthread_t client_thread_id;
  pthread_attr_t client_attr;
  void *res;
  config_t config;
  config.no_of_messages = no_of_messages;
  config.message_size = message_size;
  pthread_attr_init(&client_attr);
  pthread_create(&client_thread_id, &client_attr, client, &config);

  void *message = malloc(message_size);
  while (1) {
    connfd = accept(listenfd, (struct sockaddr*)NULL, NULL); 

    tbb::tick_count tick_start = tbb::tick_count::now();
    unsigned long bytes_received = 0;
    while (bytes_received < no_of_messages * (unsigned long)message_size) {
      bytes_received += read(connfd, message, message_size);
    }
    tbb::tick_count tick_end = tbb::tick_count::now();
    double elapsed = (tick_end - tick_start).seconds();
    printf("receiver time elapsed = %f\n", elapsed);

    close(connfd);
    break;
  }
  
  close(listenfd);
  free(message);
  pthread_join(client_thread_id, &res);
}

int main(int argc, char *argv[]) {
  int test_type = argc > 1 ? atoi(argv[1]) : 0;
  int no_of_messages = argc > 2 ? atoi(argv[2]) : 10000;
  int message_size = argc > 3 ? atoi(argv[3]) : 400;

  if (test_type == 0) {
    test_throughput(no_of_messages, message_size);
  } else {
    test_latency(no_of_messages, message_size);
  }
}