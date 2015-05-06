LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas
CFLAGS = -g -O3
INCLUDE_PATH = -I/share/probe/misc/cuda-6.5/include
LIBPATH = -L/usr/local/cuda/lib64

all: Makefile mlr.cpp
	nvcc mlr.cpp mlr-util.cu syncedmem.cpp math_functions.cpp common.cpp math_functions_cuda.cu $(LIBS) -o mlr $(CFLAGS)

mpi: Makefile mlr.cpp
	nvcc -o mlr-util.o -c mlr-util.cu $(CFLAGS)
	nvcc -o math_functions_cuda.o -c math_functions_cuda.cu $(CFLAGS)
	mpic++ -o mlr mlr-util.o math_functions_cuda.o math_functions.cpp common.cpp syncedmem.cpp mlr.cpp $(INCLUDE_PATH) $(LIBPATH) $(LIBS) $(CFLAGS)
