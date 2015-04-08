LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt

all: Makefile tick_test.cpp
	g++ tick_test.cpp $(LIBS) -o tick_test -g -O3
  
mpi: Makefile tick_test.cpp
	mpic++ tick_test.cpp $(LIBS) -o tick_test -g -O3

