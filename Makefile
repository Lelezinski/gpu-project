CXX = g++
NVCC = nvcc
SRC_DIR = ./src/
OBJ_DIR = ./obj/
INC_DIR_UTILS = $(SRC_DIR)utils/
INC_DIR_CPU = $(SRC_DIR)1_CPU_naive/
INC_DIR_GPU = $(SRC_DIR)2_GPU_naive/
INC_DIR_GPUOPT1 = $(SRC_DIR)3_GPU_opt1/
EXEC_CPU = smith_waterman_cpu.exe
EXEC_GPU = smith_waterman_gpu.exe
EXEC_GPUOPT1 = smith_waterman_gpuopt1.exe

CXXFLAGS = -Wall -g -I$(INC_DIR_UTILS)
NVCCFLAGS = -I$(INC_DIR_UTILS) -g

# Source files for CPU, single GPU, and batch GPU implementations
SRC_CPU = $(SRC_DIR)1_CPU_naive/swa.cpp $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp
SRC_GPU = $(SRC_DIR)2_GPU_naive/swa.cu $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp
SRC_GPUOPT1 = $(SRC_DIR)3_GPU_opt1/swa.cu $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp

# Object files
OBJ_CPU = $(OBJ_DIR)main.o $(OBJ_DIR)cpu_swa.o $(OBJ_DIR)utils.o
OBJ_GPU = $(OBJ_DIR)main.o $(OBJ_DIR)gpu_swa.o $(OBJ_DIR)utils.o
OBJ_GPUOPT1 = $(OBJ_DIR)main.o $(OBJ_DIR)gpuopt1_swa.o $(OBJ_DIR)utils.o

# Default target
all: cpu gpu gpuopt1

# CPU target
cpu: CXXFLAGS += -I$(INC_DIR_CPU)
cpu: $(OBJ_DIR) $(EXEC_CPU)

$(EXEC_CPU): $(OBJ_CPU)
	$(CXX) $(CXXFLAGS) -o $(EXEC_CPU) $(OBJ_CPU)

$(OBJ_DIR)cpu_swa.o: $(SRC_DIR)1_CPU_naive/swa.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Single GPU target
gpu: $(OBJ_DIR) $(EXEC_GPU)

$(EXEC_GPU): $(OBJ_GPU)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_GPU) -o $(EXEC_GPU) $(OBJ_GPU)

$(OBJ_DIR)gpu_swa.o: $(SRC_DIR)2_GPU_naive/swa.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_GPU) -c $< -o $@

# Opt1 GPU target
gpuopt1: $(OBJ_DIR) $(EXEC_GPUOPT1)

$(EXEC_GPUOPT1): $(OBJ_GPUOPT1)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_GPUOPT1) -o $(EXEC_GPUOPT1) $(OBJ_GPUOPT1)

$(OBJ_DIR)gpuopt1_swa.o: $(SRC_DIR)3_GPU_opt1/swa.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_GPUOPT1) -c $< -o $@

# Common object files for all builds
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)main.o: $(SRC_DIR)main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)utils.o: $(SRC_DIR)utils/utils.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(EXEC_CPU) $(EXEC_GPU) $(EXEC_BATCH_GPU)
