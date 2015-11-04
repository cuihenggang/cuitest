LIBS = -lglog -lgflags -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lzmq -lrt -lpthread
CFLAGS = -g -O3

all: Makefile test.cpp
	mpic++ test.cpp $(LIBS) -o test $(CFLAGS)

run:
	mpirun -machinefile hostlist ./test 1 10 536870912
