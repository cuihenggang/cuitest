LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt -lnuma

all: Makefile test.cpp
	g++ test.cpp $(LIBS) -o test -g -O3
  

