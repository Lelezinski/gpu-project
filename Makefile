CXX = g++
SRC_DIR = ./src/
INC_DIR = ./src/inc/
OBJ_DIR = ./obj/
EXEC = smith_waterman.exe

CXXFLAGS = -Wall -g -I$(INC_DIR)

SRC = $(SRC_DIR)main.cpp $(SRC_DIR)swa.cpp $(SRC_DIR)utils.cpp
OBJ = $(OBJ_DIR)main.o $(OBJ_DIR)swa.o $(OBJ_DIR)utils.o

all: $(OBJ_DIR) $(EXEC)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJ)

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(EXEC)
