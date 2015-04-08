LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt

all: Makefile test.cpp
	g++ test.cpp $(LIBS) -o test -g -pg -O3
