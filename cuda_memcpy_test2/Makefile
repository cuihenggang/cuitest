LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas
CFLAGS = -g -O3
INCLUDE_PATH = -I/usr/local/cuda/include
LIBPATH = -L/usr/local/cuda/lib64

all: Makefile test.cpp
	nvcc test.cpp $(LIBS) -o test $(CFLAGS)
