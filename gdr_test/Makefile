LIBS = -lglog -lgflags -lrt -lcudart -lcublas
CFLAGS = -g -O3
CUDA_HOME = /usr/local/cuda
MPI_HOME = /opt/mvapich2/gdr/2.1/cuda7.5/gnu
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile test.cpp
	$(MPI_HOME)/bin/mpic++ test.cpp $(INCLUDE_PATH) $(LIBPATH) $(LIBS) -o test $(CFLAGS)
