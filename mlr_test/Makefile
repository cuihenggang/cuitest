LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt -lcblas

all: Makefile mlr.cpp
	g++ -DCPU_ONLY mlr.cpp syncedmem.cpp math_functions.cpp $(LIBS) -o mlr -g -O3
