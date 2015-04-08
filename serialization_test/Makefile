LIBS = -lzmq -lboost_thread-mt -ltbb -lboost_program_options-mt -lboost_serialization -ltcmalloc_minimal -lrt

all: Makefile serialization_test.cpp
	g++ serialization_test.cpp $(LIBS) -o serialization_test -g -O3

