LIBS = -lboost_system -lboost_thread -lboost_program_options -lboost_serialization -ltbb -ltcmalloc_minimal -lrt

all: Makefile mlr.cpp
	g++ mlr.cpp $(LIBS) -o mlr -g -O3
