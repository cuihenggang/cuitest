LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt -lnuma

all: Makefile cpu_test.cpp
	g++ cpu_test.cpp $(LIBS) -o cpu_test -g -O3
  

