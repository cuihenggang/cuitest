LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas
CFLAGS = -g -O3
CUDA_HOME = /share/probe/misc/cuda-6.5
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile test.cpp
	$(CUDA_HOME)/bin/nvcc test.cpp row-op-util.cu gpu-util/caffe.cpp gpu-util/math_functions.cpp gpu-util/math_functions_cuda.cu gpu-util/syncedmem.cpp $(LIBS) -o test $(CFLAGS)
