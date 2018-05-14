# CMake config file to build the C++ Simulator
#
# For Windows, we always build static executables. QISKit provides the 
# necessary external binary libraries (.lib and .dll), these .lib come from an
# external source and they are simple import files, so we still need the .dll
# files in order to run the final executable. We only support MinGW64 toolchain
# so the static linking makes sure that other required libraries from this
# toolchain are included in the final executable. As MinGW64 can be installed
# anywhere in the system, the user needs to provide the PATH where the pthreads
# library is installed using the USER_LIB_PATH variable, like:
#     C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib -G "MinGW Makefiles" ..
#
# For Linux and Mac, we can build both statically or dynamically. The former is the
# default. If you want to build a dynamic executable, you need to set
# STATIC_LINKING to True, example:
#     out$ cmake -DSTATIC_LINKING=False ..
#
# For Mac, you probably need to install static versions of the toolchain in order
# to make a static executable. Additionaly, the OpenMP features are only supported
# on GNU g++ compiler, CLang doesn't include it so if you are building with CLang
# you won't get all the performance.

project(qasm_simulator_cpp VERSION 1.0 LANGUAGES CXX)

set(QASM_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(QASM_SIMULATOR_CPP_SRC
    "${QASM_SIMULATOR_CPP_SRC_DIR}/main.cpp")
set(QASM_SIMULATOR_CPP_EXTERNAL_LIBS
    "${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/headers"
    "${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/win64/lib"
	"${USER_LIB_PATH}")

# Target definition
add_executable(qasm_simulator_cpp ${QASM_SIMULATOR_CPP_SRC})

# Target properties: C++ program
set_target_properties(qasm_simulator_cpp PROPERTIES LINKER_LANGUAGE CXX)

# Toolchain options
set_property(TARGET qasm_simulator_cpp PROPERTY CXX_STANDARD 11)

if(STATIC_LINKING)
	# Hack: Seems like enable_cxx_compiler_flag_if_supported() is not properly
	# working on MacOS, when a flag is not supported, it cascades errors
	# to the rest of the flags being tested... and -static compilation on Mac
	# with gcc is failing...
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	    message(WARNING "Clang on MacOS doesn't support some -static-* flags. Switching to dyn compilation...")
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
	# Compiler flags
	enable_cxx_compiler_flag_if_supported("-O3")
	enable_cxx_compiler_flag_if_supported("-march=native")
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

# For header only libraries
SET(CMAKE_FIND_LIBRARY_PREFIXES "" ${CMAKE_FIND_LIBRARY_PREFIXES})
SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .hpp)

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Looking for external libraries
# This is a header-only library so we don't want it to link it, but we still
# want CMake to find it.
find_library(LIB_JSON
	NAMES json
	PATHS ${QASM_SIMULATOR_CPP_EXTERNAL_LIBS})
if(NOT LIB_JSON)
    message(FATAL_ERROR "JSON library not found!. Please provide with a USER_LIB_PATH to CMake so it can be searched there")
endif()

find_library(BLAS_LIB
	NAMES openblas mkl blas
	PATHS ${QASM_SIMULATOR_CPP_EXTERNAL_LIBS})
if(NOT BLAS_LIB)
	message(FATAL_ERROR "BLAS library not found!. Please provide with a USER_LIB_PATH to CMake so it can be searched there")
endif()

set(LIBRARIES PRIVATE ${BLAS_LIB})
				  
# Linking
target_link_libraries(qasm_simulator_cpp ${LIBRARIES})

set(QASM_SIMULATOR_CPP_OUTPUT_DIR $<TARGET_FILE_DIR:qasm_simulator_cpp>
	CACHE INTERNAL "Output directory for building QISKit C++ Simulator")

# TODO Windows 32 bits builds are not supported yet
set(ARCH_BITS "64")
if(ARCH32)
	set(ARCH_BITS "32")
endif()

# Copying necessary .dll files to output
# TODO look into "install()" CMake function
if(WIN32) # Windows 64 bits is also true here
	set(QASM_SIMULATOR_CPP_THIRD_PARTY_DLLS
		"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/win${ARCH_BITS}/dll/openblas.dll"
		CACHE INTERNAL "Third-party C++ Simulator DLLs")
	foreach(dll_file ${QASM_SIMULATOR_CPP_THIRD_PARTY_DLLS})
		add_custom_command(
			TARGET qasm_simulator_cpp
			POST_BUILD
			COMMAND ${CMAKE_COMMAND}
			ARGS -E copy ${dll_file} ${QASM_SIMULATOR_CPP_OUTPUT_DIR}/
		)
		# For 'make clean' target
		get_filename_component(FINAL_FILE ${dll_file} NAME)
		set_property(DIRECTORY APPEND PROPERTY
			ADDITIONAL_MAKE_CLEAN_FILES
				${QASM_SIMULATOR_CPP_OUTPUT_DIR}/${FINAL_FILE})
	endforeach()
endif()
	
# Tests
# TODO: Enable them when ready
#add_subdirectory(${QASM_SIMULATOR_CPP_DIR}/test)
