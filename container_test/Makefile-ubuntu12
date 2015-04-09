LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt

all: Makefile container_test.cpp
	g++ container_test.cpp $(LIBS) -o container_test -g -O3
  
mpi: Makefile container_test.cpp
	mpic++ container_test.cpp $(LIBS) -o container_test -g -O3

