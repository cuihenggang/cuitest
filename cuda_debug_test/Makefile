LIBS = -lrt -lcudart
CFLAGS = -g -O3
INCLUDE_PATH = -I/share/probe/misc/cuda-7.0/include
LIBPATH = -L/share/probe/misc/cuda-7.0/lib64

all: Makefile test.cu
	/share/probe/misc/cuda-7.0/bin/nvcc test.cu $(LIBS) -o test $(CFLAGS)
