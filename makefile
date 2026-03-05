# =========================
# Common project settings
# =========================
HEADER_DIR = build/headers
SRC_DIR= build/src
BUILD_DIR  = build

# =========================
# HIP / ROCm toolchain (AMD)
# =========================
HIP_HOME = /opt/rocm
HIPCC= hipcc

# MI300A is usually gfx942 – verify with `rocminfo`
AMDGPU_TARGET = gfx942
HIPFLAGS  = -O3 -g -std=c++17 -Wall -Wextra --offload-arch=$(AMDGPU_TARGET)
# hipBLAS/hipSPARSE wrappers over rocBLAS/rocSPARSE

HIP_LIBS  = -L$(HIP_HOME)/lib -lamdhip64 -lhipblas -lhipsparse
HIP_ARGS = $(shell /opt/rocm/bin/hipconfig --cpp_config)
CPP= g++
CFLAGS = -O3 -g -std=c++17 -Wall -Wextra -march=native $(HIP_ARGS)
DIRECTORIES = $(HEADER_DIR) $(HEADER_DIR)/coagulation $(SRC_DIR)

# =========================
# Mode switch
#   HIP_MODE=1  → HIP/ROCm
#   default → CUDA
# =========================
GPU_COMPILER = $(HIPCC)
GPU_FLAGS= $(HIPFLAGS)
GPU_INCLUDE  = -I./$(HEADER_DIR) $(HIP_ARGS)
GPU_LIBS = $(HIP_LIBS)

# =========================
# Headers and objects
# =========================
COAG_HEADERS := coagulation.h kernels.h fragments.h size_grid.h integration.h
COAG_HEADERS := $(addprefix coagulation/, $(COAG_HEADERS))
HEADERS := grid.h field.h cuda_array.h reductions.h utils.h matrix_types.h scan.h \
   stellar_irradiation.h planck.h opacity.h constants.h FLD.h FLD_device.h \
   pcg_solver.h radmc3d_utils.h star.h timing.h bins.h advection.h \
   diffusion_device.h sources.h gas1d.h DSHARP_opacs.h file_io.h errorfuncs.h \
   dustdynamics.h dustdynamics1D.h van_leer.h drag_const.h $(COAG_HEADERS)

HEADERS := $(addprefix $(HEADER_DIR)/, $(HEADERS))

OLD_OBJ := grid.o integrate_z.o scan.o scan3d.o zero_bounds.o copy.o \
   hydrostatic.o pcg_solver.o stellar_irradiation.o FLD_mono.o FLD_multi.o \
   jacobi.o ILU_precond.o gmres.o block_jacobi.o sparse_utils.o \
   radmc3d_utils.o timing.o star.o bins.o check_tol.o advection.o diffusion.o \
   coagulation.o coagulation_init.o coagulation_integrate.o super_stepping.o \
   sources.o gas1d.o DSHARP_opacs.o dustdynamics.o dustdynamics1D.o

OBJ := grid.o sources.o dustdynamics.o scan3d.o scan.o zero_bounds.o bins.o DSHARP_opacs.o \
	coagulation.o coagulation_init.o coagulation_integrate.o 


OBJ := $(addprefix $(BUILD_DIR)/, $(OBJ))

TESTS_CPP = $(wildcard tests/codes/test_*.cpp)
TESTS_CU  = $(wildcard tests/codes/test_*.cu)
UNITS = $(wildcard unit_tests/unit_*.cpp)

TEST_OBJ = \
$(patsubst tests/codes/%.cpp,%, $(TESTS_CPP)) \
$(patsubst tests/codes/%.cu,%, $(TESTS_CU))

UNIT_TESTS = $(patsubst unit_tests/%.cpp,%,$(UNITS))

LIBRARY= lib/libcudisc.a

.PHONY: tests clean bintidy lib run_units cuda_build hip_build all
.SECONDARY: $(HEADERS) $(DIRECTORIES) $(OBJ)

all: cuda_build

# High-level targets

cuda_build: HIP_MODE=
cuda_build: $(LIBRARY)

hip_build: HIP_MODE=1
hip_build: $(LIBRARY)

tests: $(TEST_OBJ)
lib: $(LIBRARY)

$(LIBRARY): $(OBJ)
	@mkdir -p $(dir $@)
	ar -rcs $@ $(OBJ)

# =========================
# Compilation rules
# =========================

$(HEADER_DIR)/%.h: headers/%.h
	@mkdir -p $(DIRECTORIES)
	hipify-perl $< > $@
$(SRC_DIR)/%.cu: src/%.cu
	@mkdir -p $(DIRECTORIES)
	hipify-perl $< > $@
$(SRC_DIR)/%.cpp: src/%.cpp
	@mkdir -p $(DIRECTORIES)
	hipify-perl $< > $@

# Host C++ sources (always from src/, same in CUDA and HIP builds)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) makefile
	@mkdir -p $(BUILD_DIR)
	$(CPP) $(CFLAGS) -I./$(HEADER_DIR) -c $< -o $@

# GPU sources: .cu – from src/ in CUDA mode, from hip_src/ in HIP mode
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) makefile
	@mkdir -p $(BUILD_DIR)
	$(GPU_COMPILER) $(GPU_FLAGS) $(GPU_INCLUDE) -c $< -o $@
# =========================
# Test binaries
# =========================
# CPU-only tests (.cpp)
test_%: $(PWD)/tests/codes/test_%.cpp $(LIBRARY) $(HEADERS) makefile
	$(CPP) $(CFLAGS) -I./$(HEADER_DIR) $< -o $@ $(LIBRARY) $(GPU_LIBS)

# GPU tests (.cu) – from tests/codes in CUDA, from hip_tests in HIP
test_%: $(PWD)/$(TEST_GPU_DIR)/test_%.cu $(LIBRARY) $(HEADERS) makefile
	$(GPU_COMPILER) $(GPU_FLAGS) $(GPU_INCLUDE) $< -o $@ $(LIBRARY) $(GPU_LIBS)

# Standalone codes in codes/
%: codes/%.cpp $(LIBRARY) $(HEADERS) makefile
	$(CPP) $(CFLAGS) -I./$(HEADER_DIR) $< -o $@ $(LIBRARY) $(GPU_LIBS)

# Unit tests (CPU binaries, but link with GPU libs so HIP/CUDA symbols resolve)
unit_%: unit_tests/unit_%.cpp $(LIBRARY) $(HEADERS) makefile
	$(CPP) $(CFLAGS) -I./$(HEADER_DIR) $< -o $@ $(LIBRARY) $(GPU_LIBS)

run_units: $(UNIT_TESTS)
	@for executable in $(UNIT_TESTS); do \
		if [ -x "$$executable" ]; then \
		./$$executable; \
		wait; \
		fi; \
	done
	clean:
	rm -rf build/*.o $(TEST_OBJ) $(LIBRARY)

bintidy:
	rm -f ./test_* unit_adv_diff unit_coag unit_temp