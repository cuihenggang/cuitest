LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas
CFLAGS = -g -O3
INCLUDE_PATH = -I/share/probe/misc/cuda-6.5/include
LIBPATH = -L/share/probe/misc/cuda-6.5/lib64

all: Makefile test.cu
	/share/probe/misc/cuda-6.5/bin/nvcc test.cu $(LIBS) -o test $(CFLAGS)
