INCLUDE = -I./ -I./backends -I./engines -I./utilities -I./third-party -I./third-party/headers
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual
OPT = -O3 -march=native -ffast-math
# Install exe into pip location for backends
OUTPUT_DIR = ../../../qiskit/backends

CC = g++

# Use Homebrew GNU g++ for enabling OpenMP on macOS

OS:=$(shell uname)
GCC7:=$(shell g++-7 --version | grep ^'command not found')
ifeq ($(OS)$(GCC7),Darwin)
	CC = g++-7
endif

# Check if compiler is Apple LLVM and doesn't support OpenMP
APPLELLVM = $(shell ${CC} --version | grep ^Apple)
ifeq ($(APPLELLVM),)
	CPPFLAGS = $(USER_FLAGS) -std=c++11 -fopenmp $(OPT) $(INCLUDE) $(WARNINGS)
else
	CPPFLAGS = $(USER_FLAGS) -std=c++11 $(OPT) $(INCLUDE) $(WARNINGS)
endif

# BLAS distribution to use (change to use OpenBLAS or MKL for example)
ifeq ($(OS),Darwin)
	LIB_BLAS = -framework Accelerate
else
	LIB_BLAS = -llapack -lblas
endif
LIBS = -lpthread $(LIB_BLAS)

.SUFFIXES:.cpp .cc .o .c
.cpp.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o $@ $<
.cc.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o $@ $<
.c.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o $@ $<


all: sim
debug: sim_debug

sim: main.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qasm_simulator_cpp main.o $(LIBS)

sim_debug: main.o
	$(CC) -g $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qasm_simulator_cpp_debug main.o $(LIBS)

clean:
	rm -rf ${OUTPUT_DIR}/qasm_simulator_cpp main.o

