LIBS = -lrt -lcudart -lglog
CFLAGS = -g -O3
CUDA_HOME = /usr/local/cuda
INCLUDE_PATH = -I$(CUDA_HOME)/include
LIBPATH = -L$(CUDA_HOME)/lib64

all: Makefile test.cu
	$(CUDA_HOME)/bin/nvcc test.cu $(LIBS) -o test $(CFLAGS)
