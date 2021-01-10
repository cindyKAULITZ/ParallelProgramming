CC=nvcc -c 
CC_FLAGS=-std=c++11 -O3 -Xptxas="-v" --ptxas-options=-v -arch=sm_61 -lm 
# CC_FLAGS=-std=c++11 -Wall -g -O3  -DDEBUG_KNN



# File names
EXEC = bin/main
EXECOBJ = $(EXEC:bin/%=obj/%.o)
SOURCES = $(wildcard src/*.cpp)
CUSOURCES = $(wildcard src/*.cu)
INCLUDES = $(wildcard src/*.h)
OBJECTS = $(SOURCES:src/%.cpp=obj/%.o)
CUOBJECTS = $(CUSOURCES:src/%.cu=obj/%.o)

TESTSOURCES = $(wildcard test/*.cpp)
TESTOBJECTS = $(filter-out $(EXECOBJ), $(OBJECTS))
TESTTARGETS = $(TESTSOURCES:test/%.cpp=bin/%)

all: $(EXEC)

test: $(TESTTARGETS)

# Main target
$(EXEC): $(OBJECTS) $(CUOBJECTS)
	g++ $(OBJECTS) $(CUOBJECTS)  -L/opt/cuda/lib64/ -lcudart -o $(EXEC)

$(TESTTARGETS): bin/% : test/%.cpp $(TESTOBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $< $(TESTOBJECTS) -Isrc/
#	./$@

# # To obtain object files
$(OBJECTS): obj/%.o : src/%.cpp
	g++ -c -std=c++11 -Wall -g -O3  -DDEBUG_KNN $< -o $@

$(CUOBJECTS): obj/%.o : src/%.cu
	$(CC) -c $(CC_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS) $(CUOBJECTS) $(TESTTARGETS)
