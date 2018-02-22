# Default values for global scope variables.
# They can be overriden by passing -DVARIABLE=Value to cmake, like:
#     out$ cmake -DSTATIC_LINKING=False ..
#

set(CMAKE_BUILD_TYPE "Release")
set(STATIC_LINKING True CACHE BOOL "Static linking of executables")
set(ENABLE_TARGETS_NON_PYTHON True CACHE BOOL "Enable targets for non Python code")
set(ENABLE_TARGETS_QA True CACHE BOOL "Enable targets for QA targets")
