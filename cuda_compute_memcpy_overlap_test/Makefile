LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas
CFLAGS = -g -O3
INCLUDE_PATH = -I/share/probe/misc/cuda-6.5/include
LIBPATH = -L/usr/local/cuda/lib64
#CC = nvcc --default-stream per-thread
CC = nvcc

all: Makefile test.cpp
	$(CC) test.cpp $(LIBS) -o test $(CFLAGS)
