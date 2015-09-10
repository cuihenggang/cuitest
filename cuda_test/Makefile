LIBS = -lglog -lgflags -lboost_system -lcudart -lcublas -lcurand -ltbb
CUDA_HOME = /share/probe/misc/cuda-6.5
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile cuda_speed_test.cpp math_functions.cu common.cpp syncedmem.cpp
	/share/probe/misc/cuda-6.5/bin/nvcc cuda_speed_test.cpp math_functions.cu common.cpp syncedmem.cpp $(INCLUDE_PATH) $(LIBPATH) $(LIBS) -o cuda_speed_test -g -O3

test: Makefile cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp
	/share/probe/misc/cuda-6.5/bin/nvcc cuda_test.cpp math_functions.cu common.cpp syncedmem.cpp $(INCLUDE_PATH) $(LIBPATH) $(LIBS) -o cuda_test -g -O3