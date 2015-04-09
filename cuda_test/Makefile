LIBS = -lglog -lgflags -lboost_system -lcudart -lcublas -lcurand -ltbb
INCLUDES = -I/usr/local/cuda/include

all: Makefile cuda_speed_test.cpp math_functions.cu common.cpp syncedmem.cpp
	nvcc cuda_speed_test.cpp math_functions.cu common.cpp syncedmem.cpp $(LIBS) -o cuda_speed_test -g -O3

test: Makefile cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp
	nvcc cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp $(LIBS) -o cuda_test -g -O3