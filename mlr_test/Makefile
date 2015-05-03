LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas -lcudart -lcublas

all: Makefile mlr.cpp
	nvcc -DCPU_WORKER mlr.cpp syncedmem.cpp math_functions.cpp common.cpp math_functions.cu $(LIBS) -o mlr -g -O3
