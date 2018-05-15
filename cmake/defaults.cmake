# Default values for global scope variables.
# They can be overriden by passing -DVARIABLE=Value to cmake, like:
#     out$ cmake -DSTATIC_LINKING=True ..
#

set(CMAKE_BUILD_TYPE "Release")
set(STATIC_LINKING False CACHE BOOL "Static linking of executables")
set(ENABLE_TARGETS_NON_PYTHON True CACHE BOOL "Enable targets for non Python code")
set(ENABLE_TARGETS_QA True CACHE BOOL "Enable targets for QA targets")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH64 True CACHE BOOL "We are on a 64 bits platform")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ARCH32 True CACHE BOOL "We are on a 32 bits platform")
endif()