LIBS = -lglog -lgflags -lboost_system -lcudart -lcublas -lcurand

all: Makefile cuda_test.cpp math_functions.cu common.cpp
	nvcc cuda_test.cpp math_functions.cu common.cpp $(LIBS) -o cuda_test -g -O3
