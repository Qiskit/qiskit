
CC = g++
#CC = g++-7 # use Homebrew GNU g++ for enabling OpenMP on macOS

INCLUDE = -I./ -I./backends -I./engines -I./utilities -I./third-party
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual
OPT = -O3 -march=native -ffast-math

# Check if compiler is Apple LLVM and doesn't support OpenMP
APPLELLVM = $(shell ${CC} --version | grep ^Apple)
ifeq ($(APPLELLVM),)
	CPPFLAGS = -std=c++14 -fopenmp $(OPT) $(INCLUDE) $(WARNINGS)
else
	CPPFLAGS = -std=c++14 $(OPT) $(INCLUDE) $(WARNINGS)
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


all: sim

sim: main.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ../local_qiskit_simulator main.o $(LIBS)

clean:
	rm -f *.o ../local_qiskit_simulator main.o *.*~ *~ 2>/dev/null

