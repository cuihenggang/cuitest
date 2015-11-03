#include <iostream>
#include <assert.h>

#include <glog/logging.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <mpi.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

using namespace std;

int main (int argc, char *argv[])
{
  int myid, numprocs, i;
  int size;
  MPI_Status reqstat;
  char *s_buf, *r_buf;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  double t_start = 0.0, t_end = 0.0;

  size = 1 << 22;
  CUDA_CHECK(cudaMalloc(&s_buf, size));
  CUDA_CHECK(cudaMalloc(&r_buf, size));

  MPI_Barrier(MPI_COMM_WORLD);

  if(myid == 0) {
    MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
    MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &reqstat);
  }

  else if(myid == 1) {
    MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &reqstat);
    MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
  }

  // if(myid == 0) {
      // double latency = (t_end - t_start) * 1e6 / (2.0 * options.loop);

      // fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
              // FLOAT_PRECISION, latency);
      // fflush(stdout);
  // }

  CUDA_CHECK(cudaFree(s_buf));
  CUDA_CHECK(cudaFree(r_buf));
  MPI_Finalize();
  return 0;
}

