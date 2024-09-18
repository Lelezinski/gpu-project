CXX = g++
NVCC = nvcc
SRC_DIR = ./src/
OBJ_DIR = ./obj/
INC_DIR_UTILS = $(SRC_DIR)utils/
INC_DIR_CPU = $(SRC_DIR)1_CPU_naive/
INC_DIR_GPU = $(SRC_DIR)2_GPU_naive/
INC_DIR_BATCH_GPU = $(SRC_DIR)4_GPU_batch/
EXEC_CPU = smith_waterman_cpu.exe
EXEC_GPU = smith_waterman_gpu.exe
EXEC_BATCH_GPU = smith_waterman_batch_gpu.exe

CXXFLAGS = -Wall -g -I$(INC_DIR_UTILS)
NVCCFLAGS = -I$(INC_DIR_UTILS) -g

# Source files for CPU, single GPU, and batch GPU implementations
SRC_CPU = $(SRC_DIR)1_CPU_naive/swa.cpp $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp
SRC_GPU = $(SRC_DIR)2_GPU_naive/swa.cu $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp
SRC_BATCH_GPU = $(SRC_DIR)4_GPU_batch/swa.cu $(SRC_DIR)main.cpp $(SRC_DIR)utils/utils.cpp

# Object files
OBJ_CPU = $(OBJ_DIR)main.o $(OBJ_DIR)cpu_swa.o $(OBJ_DIR)utils.o
OBJ_GPU = $(OBJ_DIR)main.o $(OBJ_DIR)gpu_swa.o $(OBJ_DIR)utils.o
OBJ_BATCH_GPU = $(OBJ_DIR)main.o $(OBJ_DIR)batch_gpu_swa.o $(OBJ_DIR)utils.o

# Default target
all: cpu gpu

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

# Batch GPU target
batch_gpu: $(OBJ_DIR) $(EXEC_BATCH_GPU)

$(EXEC_BATCH_GPU): $(OBJ_BATCH_GPU)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_BATCH_GPU) -o $(EXEC_BATCH_GPU) $(OBJ_BATCH_GPU)

$(OBJ_DIR)batch_gpu_swa.o: $(SRC_DIR)4_GPU_batch/swa.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR_BATCH_GPU) -c $< -o $@

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
