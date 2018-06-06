# Default values for global scope variables.
# They can be overriden by passing -DVARIABLE=Value to cmake, like:
#     out$ cmake -DSTATIC_LINKING=True ..
#

# Set the build type: Release or Debug
set(CMAKE_BUILD_TYPE "Release")

# Set the type of linking we want to make for native code. Static builds are
# great for distributing. Apple doesn't allow full static linking, so only user
# libraries will be included in the final binary.
set(STATIC_LINKING False CACHE BOOL "Static linking of executables")

# Enable or disable CMake targets that depends on native code.
set(ENABLE_TARGETS_NON_PYTHON True CACHE BOOL "Enable targets for non Python code")

# Enable or disable CMake targets for Q&A: tests/linter/style
set(ENABLE_TARGETS_QA True CACHE BOOL "Enable targets for QA targets")

# Detect platform bit size
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH64 True CACHE BOOL "We are on a 64 bits platform")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ARCH32 True CACHE BOOL "We are on a 32 bits platform")
endif()
