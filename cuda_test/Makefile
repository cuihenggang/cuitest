LIBS = -lglog -lgflags -lboost_system -lcudart -lcublas -lcurand
INCLUDES = -I/usr/local/cuda/include

all: Makefile cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp
	nvcc cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp $(LIBS) -o cuda_test -g -O3
