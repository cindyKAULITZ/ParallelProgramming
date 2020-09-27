CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O2
CFLAGS = -O3
TARGETS = lab1_Q
  
.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
