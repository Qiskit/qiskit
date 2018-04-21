INCLUDE = -I./ -I./backends -I./engines -I./utilities -I./third-party -I./third-party/headers
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual
OPT = -O3 -march=native -ffast-math
OUTPUT_DIR = ../../../out/src/qasm-simulator-cpp

CC = g++

# Use Homebrew GNU g++ for enabling OpenMP on macOS
OS:=$(shell uname)
GCC7:=$(shell g++-7 --version | grep ^'command not found')
ifeq ($(OS)$(GCC7),Darwin)
	CC = g++-7
endif

# -march not supported on ppc64le (Power). Use -mcpu instead
ARCH:=$(shell uname -m)
ifeq ($(ARCH),ppc64le)
	OPT = -O3 -mcpu=native -ffast-math
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
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<
.cc.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<
.c.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<


all: directories sim
debug: directories sim_debug

directories:
	mkdir -p $(OUTPUT_DIR)

sim: main.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qasm_simulator_cpp ${OUTPUT_DIR}/main.o $(LIBS)

sim_debug: main.o
	$(CC) -g $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qasm_simulator_cpp_debug ${OUTPUT_DIR}/main.o $(LIBS)

depend: 
	../build_dependencies.sh

clean:
	rm -rf ${OUTPUT_DIR}

