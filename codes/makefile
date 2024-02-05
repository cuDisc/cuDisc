# Example makefile for compilation with cuDisc as a static library. 
# Change lines 4 and 5 to the correct paths to both your cuda installation and cuDisc location.

CUDA_HOME = /path/to/cuda-12.0
CUDISC_HOME = /path/to/cuDisc

ARCH=--generate-code arch=compute_60,code=sm_60 \
	--generate-code arch=compute_61,code=sm_61 \
	--generate-code arch=compute_62,code=sm_62 \
	--generate-code arch=compute_70,code=sm_70 \
	--generate-code arch=compute_72,code=sm_72 \
	--generate-code arch=compute_75,code=sm_75 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_86,code=sm_86 

CUDA = nvcc 
CUDAFLAGS = -O3 -g --std=c++17 -Wno-deprecated-gpu-targets $(ARCH)

CPP = g++  
CFLAGS = -O3 -g  -std=c++17 -Wall -Wextra -march=native 

INCLUDE = -I$(CUDISC_HOME)/headers -I$(CUDA_HOME)/include

LIB = -L$(CUDISC_HOME)/lib -lcudisc \
	-L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse

all : cuda_file cpp_file
.PHONY : tidy

cuda_file: cuda_file.cu makefile				#replace "cuda_file" with name of the cuda file you wish to compile
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $< -o $@  $(LIB)

cpp_file :  cpp_file.cpp makefile 				#replace "cpp_file" with name of the cpp file you wish to compile
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@  $(LIB)

clean:
	rm -rf cuda_file cpp_file

tidy:
	rm -rf outputs
