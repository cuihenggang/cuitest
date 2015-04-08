// #include <mpi.h>

#include <sys/time.h>

#include <iostream>
#include <stdlib.h>

#include <tbb/tick_count.h>

int main(int argc, char* argv[])
{
  // MPI_Init(&argc, &argv);
  
  int count = atoi(argv[1]);
  tbb::tick_count tick_start;
  tbb::tick_count tick_end;
  // int rank;
  double total_time = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tick_start = tbb::tick_count::now();
  for (int i = 0; i < count; i ++) {
    tick_end = tbb::tick_count::now();
    total_time += (tick_end - tick_start).seconds();
    tick_start = tick_end;
  }
  std::cout << "total_time = " << total_time << std::endl;
  
  // MPI_Finalize();
}
