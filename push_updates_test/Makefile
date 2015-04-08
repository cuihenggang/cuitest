LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt -lnuma

all: Makefile push_updates_test.cpp
	g++ push_updates_test.cpp $(LIBS) -o push_updates_test -g -O3
  

