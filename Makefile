CXX = g++
SRC = ./src/SW.cpp
INC = ./src/inc/
REP = ./reports/

CXXFLAGS = -fsanitize=address -g -Wall -fexceptions -I$(INC)
LDFLAGS = -fsanitize=address  

OBJ = $(SRC:.cpp=.o)
EXEC = swa

all: $(EXEC)

$(EXEC): $(OBJ)
	@mkdir -p $(REP)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ) $(EXEC)
