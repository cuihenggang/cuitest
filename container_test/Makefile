LIBS = -lzmq -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt

all: Makefile container_test.cpp
	g++ container_test.cpp $(LIBS) -o container_test -g -O3
  
mpi: Makefile container_test.cpp
	mpic++ container_test.cpp $(LIBS) -o container_test -g -O3

