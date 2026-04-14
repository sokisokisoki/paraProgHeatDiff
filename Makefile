# Heat diffusion: serial CPU, MPI+CPU, MPI+CUDA (AiMOS / V100: sm_70)
# Override on the command line, e.g. `make MPICC=mpicc CUDA_ARCH=sm_70`

MPICC       ?= mpicc
MPICXX      ?= mpicxx
CC          ?= gcc
NVCC        ?= nvcc
CUDA_ARCH   ?= sm_70
CUDA_GENCODE ?= arch=compute_70,code=sm_70

CFLAGS   := -O3 -std=c11 -Wall -Wextra -Iinclude
CXXFLAGS := -O3 -std=c++14 -Wall -Wextra -Iinclude
NVCFLAGS := -O3 -Iinclude -Xcompiler -Wall,-Wextra
LDFLAGS  :=
LIBS     :=

BUILD := build

.PHONY: all clean serial mpi_cpu mpi_cuda

all: serial mpi_cpu mpi_cuda

serial: $(BUILD)/heat_serial

mpi_cpu: $(BUILD)/heat_mpi_cpu

mpi_cuda: $(BUILD)/heat_mpi_cuda

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/heat_serial: src/heat_serial.c include/heat_common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/heat_serial.c $(LDFLAGS) $(LIBS)

$(BUILD)/heat_mpi_cpu: src/heat_mpi_cpu.c include/heat_common.h include/clockcycle.h | $(BUILD)
	$(MPICC) $(CFLAGS) -o $@ src/heat_mpi_cpu.c $(LDFLAGS) -lm $(LIBS)

$(BUILD)/kernels.o: src/kernels.cu include/kernels.cuh | $(BUILD)
	$(NVCC) $(NVCFLAGS) -gencode $(CUDA_GENCODE) -c src/kernels.cu -o $@

$(BUILD)/heat_mpi_cuda.o: src/heat_mpi_cuda.cu include/heat_common.h include/clockcycle.h include/kernels.cuh | $(BUILD)
	$(NVCC) $(NVCFLAGS) -gencode $(CUDA_GENCODE) -ccbin $(MPICXX) -c src/heat_mpi_cuda.cu -o $@

$(BUILD)/heat_mpi_cuda: $(BUILD)/heat_mpi_cuda.o $(BUILD)/kernels.o | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -o $@ $(BUILD)/heat_mpi_cuda.o $(BUILD)/kernels.o -lcudart $(LDFLAGS)

clean:
	rm -rf $(BUILD)
