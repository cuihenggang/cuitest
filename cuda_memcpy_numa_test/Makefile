LIBS = -lglog -lgflags -lboost_system -lrt -lcblas -lcudart -lcublas -lnuma
CFLAGS = -g -O3
INCLUDE_PATH = -I/usr/local/cuda/include
LIBPATH = -L/usr/local/cuda/lib64

all: Makefile test.cpp
	nvcc test.cpp $(LIBS) -o test $(CFLAGS)
