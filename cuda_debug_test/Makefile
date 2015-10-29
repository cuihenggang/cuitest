LIBS = -lrt -lcudart -lglog
CFLAGS = -g -O3
#CUDA_HOME = /share/probe/misc/cuda-6.5
CUDA_HOME = /usr/local/cuda-7.5
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile test.cu
	$(CUDA_HOME)/bin/nvcc test.cu $(LIBS) -o test $(CFLAGS)
