
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

# Static linking is the default
if(NOT STATIC_LINKING)
	set(STATIC_LINKING True CACHE BOOL "Static linking of executables")
endif()