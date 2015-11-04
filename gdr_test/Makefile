LIBS = -lglog -lgflags -lrt -lcudart -lcublas
CFLAGS = -g -O3
CUDA_HOME = /usr/local/cuda
MPI_HOME = /usr/local/mvapich2-gdr
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile test.cpp
	$(MPI_HOME)/bin/mpic++ test.cpp $(INCLUDE_PATH) $(LIBPATH) $(LIBS) -o test $(CFLAGS)

run:
	export MV2_CPU_MAPPING=0 && export MV2_USE_CUDA=1 && export MV2_USE_GPUDIRECT=1 && export MV2_GPUDIRECT_GDRCOPY_LIB=$(MPI_HOME)/gdrcopy-master/libgdrapi.so && $(MPI_HOME)/bin/mpirun_rsh -export -np 2 h0-dib h1-dib ./test
