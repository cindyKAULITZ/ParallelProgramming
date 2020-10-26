CC = gcc
CXX = g++
CFLAGS = -lm -O3
lab3_pthread: CFLAGS += -pthread
lab3_omp: CFLAGS += -fopenmp
lab3_hybrid: CC = mpicc
lab3_hybrid: CXX = mpicxx
lab3_hybrid: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = lab3_pthread lab3_omp lab3_hybrid
TARGETS_P = lab3_pthread
TARGETS_O = lab3_omp

.PHONY: all
all: $(TARGETS)

.PHONY: p
p: $(TARGETS_P)

.PHONY: o
o: $(TARGETS_O)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)