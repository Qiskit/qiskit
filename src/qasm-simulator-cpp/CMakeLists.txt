# CMake config file to build the C++ Simulator

cmake_minimum_required(VERSION 3.6)
project(qasm_simulator_cpp LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(STATIC_LINKING "Specify if we want statically link the executable (for
						redistribution mainly)" FALSE)
option(BUILD_TESTS "Specify whether we want to build tests or not" TRUE)

include(toolchain-utils)

set(QASM_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(QASM_SIMULATOR_CPP_SRC
    "${QASM_SIMULATOR_CPP_SRC_DIR}/main.cpp")
set(QASM_SIMULATOR_CPP_EXTERNAL_LIBS
    "${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/headers"
	"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/win64/lib"
	"${USER_LIB_PATH}")

# Adding support for CCache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Set default build type to Release with Debug Symbols
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: Debug Release"
		FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Target definition
add_executable(qasm_simulator_cpp ${QASM_SIMULATOR_CPP_SRC})

# Target properties: C++ program
set_target_properties(qasm_simulator_cpp PROPERTIES
	LINKER_LANGUAGE CXX
	CXX_STANDARD 14)

set_target_properties(qasm_simulator_cpp PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
	RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# In order to build for MacOSX 10.7 and above with Clang, we need to force the "deployment target" to 10.7
	# and force using libc++ instead of the default for this target: libstdc++ otherwise we could not
	# use C++11/14
	message(STATUS "Apple clang detected. Forcing OSX deployment target to: 10.7 and building with libc++")
	enable_cxx_compiler_flag_if_supported("-stdlib=libc++")
endif()

if(STATIC_LINKING)
	enable_cxx_compiler_flag_if_supported("-Woverloaded-virtual")
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(WARNING "Clang on MacOS doesn't support some -static-* flags. Switching to dyn compilation...")
		unset(STATIC_LINKING)
	else()
	    # MacOS compilers don't support -static flag either
	    if(NOT APPLE)
	        enable_cxx_compiler_flag_if_supported("-static")
	    endif()
	    # This is enough to build a semi-static executable on Mac
	    enable_cxx_compiler_flag_if_supported("-static-libgcc")
	    enable_cxx_compiler_flag_if_supported("-static-libstdc++")
	endif()
endif()

if(NOT MSVC)
	# Warnings and Errors
	enable_cxx_compiler_flag_if_supported("-pedantic")
	enable_cxx_compiler_flag_if_supported("-Wall")
	enable_cxx_compiler_flag_if_supported("-Wfloat-equal")
	enable_cxx_compiler_flag_if_supported("-Wundef")
	enable_cxx_compiler_flag_if_supported("-Wcast-align")
	enable_cxx_compiler_flag_if_supported("-Wwrite-strings")
	enable_cxx_compiler_flag_if_supported("-Wmissing-declarations")
	enable_cxx_compiler_flag_if_supported("-Wredundant-decls")
	enable_cxx_compiler_flag_if_supported("-Wshadow")
	enable_cxx_compiler_flag_if_supported("-Woverloaded-virtual")
endif()

target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR})
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/backends)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/engines)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/utilities)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/headers)

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32 OR APPLE)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

# Looking for external libraries
find_package(OpenMP QUIET)
# This is a hack for building with Apple's LLVM, which doesn't support OpenMP yet
# so we need to link with an external library: libomp.
# NOTE: CMake >= 3.12.0 doesn't need this hack. It will just find OpenMP in the
# first find_package(OpenMP) call
if(NOT "${OpenMP_FOUND}" OR NOT "${OpenMP_CXX_FOUND}")
	find_program(BREW NAMES brew)
	if(BREW)
		execute_process(COMMAND ${BREW} ls libomp RESULT_VARIABLE BREW_RESULT_CODE OUTPUT_QUIET ERROR_QUIET)
		if(BREW_RESULT_CODE)
			message(STATUS "This program supports OpenMP on Mac through Brew. Please run \"brew install libomp\"")
		else()
		execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
		set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
		set(OpenMP_CXX_LIB_NAMES "omp")
		set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
		include_directories("${BREW_LIBOMP_PREFIX}/include")
		message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")

		set(OPENMP_EXTERNAL_LIB "${OpenMP_omp_LIBRARY}")
		set(OPENMP_FOUND TRUE)
		endif()
	else()
		message(STATUS "This program supports OpenMP on Mac through Homebrew, installing Homebrew recommmended https://brew.sh")
	endif()
endif()

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	message("OpenMP found!")
endif()

set(NLOHMANN_JSON_PATH ${QASM_SIMULATOR_CPP_EXTERNAL_LIBS})
find_package(nlohmann_json REQUIRED)

find_package(Threads)

if(STATIC_LINKING)
    message(STATUS "Setting BLA_STATIC")
	set(BLA_STATIC TRUE)
endif()

if(APPLE)
    message(STATUS "Looking for Apple BLAS library...")
    set(BLA_VENDOR "Apple")
else()
    message(STATUS "Looking for OpenBLAS library...")
    set(BLA_VENDOR "OpenBLAS")
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
	message(STATUS "OpenBLAS not found. Looking for any other BLAS library...")
	unset(BLA_VENDOR)
	find_package(BLAS REQUIRED)
endif()

# Set dependent libraries
set(LIBRARIES
        ${OPENMP_EXTERNAL_LIB}
        ${BLAS_LIBRARIES}
        nlohmann_json
        Threads::Threads)

target_link_libraries(qasm_simulator_cpp ${LIBRARIES})

# Installing
find_package(PythonExtensions REQUIRED)
install(TARGETS qasm_simulator_cpp RUNTIME DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR})
