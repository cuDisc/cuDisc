

HEADER_DIR = headers
SRC_DIR = src
BUILD_DIR = build
CUDA_HOME = /usr/local/cuda-12.0

CPP = g++  
CFLAGS = -O3 -g  -std=c++17 -Wall -Wextra -march=native 

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
INCLUDE = -I./$(HEADER_DIR) -I$(CUDA_HOME)/include

LIB = -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse

COAG_HEADERS := coagulation.h kernels.h fragments.h size_grid.h integration.h
COAG_HEADERS := $(addprefix coagulation/, $(COAG_HEADERS))

HEADERS := grid.h field.h cuda_array.h reductions.h utils.h matrix_types.h \
	stellar_irradiation.h planck.h opacity.h constants.h FLD.h  FLD_device.h \
	pcg_solver.h radmc3d_utils.h star.h timing.h bins.h advection.h \
	diffusion_device.h sources.h gas1d.h DSHARP_opacs.h file_io.h errorfuncs.h \
	dustdynamics.h $(COAG_HEADERS)


OBJ := grid.o integrate_z.o scan.o scan3d.o zero_bounds.o copy.o \
	hydrostatic.o pcg_solver.o stellar_irradiation.o FLD_mono.o FLD_multi.o \
	FLD_sts.o jacobi.o ILU_precond.o gmres.o block_jacobi.o sparse_utils.o \
	radmc3d_utils.o timing.o star.o bins.o check_tol.o advection.o diffusion.o \
	coagulation.o coagulation_init.o coagulation_integrate.o  super_stepping.o \
	sources.o gas1d.o DSHARP_opacs.o dustdynamics.o

OBJ := $(addprefix $(BUILD_DIR)/, $(OBJ))
HEADERS := $(addprefix $(HEADER_DIR)/, $(HEADERS))

TESTS_CPP = $(wildcard tests/test_*.cpp)
TESTS_CU =  $(wildcard tests/test_*.cu)

TEST_OBJ = \
	$(patsubst tests/%.cpp,%, $(TESTS_CPP)) \
	$(patsubst tests/%.cu,%, $(TEST_CU))

LIBRARY = lib/libcudisc.a

.PHONY: tests clean tidy lib

tests : $(TEST_OBJ)

lib : $(LIBRARY)

$(LIBRARY): $(OBJ)
	ar -rcs $@ $(OBJ)

$(BUILD_DIR)/%.o: src/%.cpp $(HEADERS) makefile
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu  $(HEADERS) makefile
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) -c $< -o $@

test_%: tests/codes/test_%.cpp $(LIBRARY) $(HEADERS) makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@ $(LIBRARY) $(LIB)

test_%: tests/codes/test_%.cu $(LIBRARY) $(HEADERS) makefile 
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $< -o $@ $(LIBRARY) $(LIB)

%: codes/%.cpp $(LIBRARY) $(HEADERS) makefile 
	$(CPP) $(CFLAGS) $(INCLUDE)  $< -o $@ $(LIBRARY) $(LIB) 

clean:
	rm -rf build/*.o $(TEST_OBJ) $(LIBRARY)

bintidy:
	rm ./test_*
