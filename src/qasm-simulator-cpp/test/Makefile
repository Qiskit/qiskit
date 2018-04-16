
CC = g++
#CC = g++-7 # use Homebrew GNU g++ for enabling OpenMP on macOS

INCLUDE = -I../src/utilities/
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wredundant-decls -Wshadow 
OPT = -O3 -march=native -ffast-math

# Check if compiler is Apple LLVM and doesn't support OpenMP
APPLELLVM = $(shell ${CC} --version | grep ^Apple)
ifeq ($(APPLELLVM),)
	CPPFLAGS = $(USER_FLAGS) -std=c++11 -fopenmp $(OPT) $(INCLUDE) $(WARNINGS)
else
	CPPFLAGS = $(USER_FLAGS) -std=c++11 $(OPT) $(INCLUDE) $(WARNINGS)
endif

# BLAS distribution to use (change to use OpenBLAS or MKL for example)
OS:=$(shell uname)
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


all: test_qubit_vector

test_qubit_vector: test_qubit_vector.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o test_qubit_vector test_qubit_vector.o $(LIBS)

sim_debug: main.o
	$(CC) -g $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qiskit_simulator_debug ${OUTPUT_DIR}/main.o $(LIBS)

clean:
	rm -rf *.o test_qubit_vector
