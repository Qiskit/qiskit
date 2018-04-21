# Tests
enable_testing()

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/out/${CMAKE_BUILD_TYPE}/qasm_simulator_cpp/test/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# TODO Put more descriptive test names
set(test_list one two)

foreach(test_item ${test_list})
  set(test_name ${test_item}_tests)
  add_executable(${test_name} main.cpp ${test_item}_test.cpp ${headers})
  target_include_directories(${test_name} PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR})
  target_include_directories(${test_name} PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/backends)
  target_include_directories(${test_name} PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/engines)
  target_include_directories(${test_name} PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/utilities)
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  # set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 11)
  target_link_libraries(${test_name} ${QASM_SIMULATOR_CPP_EXTERNAL_LIBRARIES})
  add_test(NAME ${test_name} COMMAND ${test_name})

  if(NOT CMAKE_GENERATOR MATCHES "^Mingw")
    add_custom_target(${test_item}_test
      COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure -R ${test_name})
    add_custom_target(${test_item}_check
      COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure -R ${test_name}
      DEPENDS ${test_name})
  endif()
endforeach()

# Toolchain options
# Compiler flags
_enable_cxx_compiler_flag_if_supported("-O3")
_enable_cxx_compiler_flag_if_supported("-march=native")
_enable_cxx_compiler_flag_if_supported("-fopenmp")

# Warnings and Errors
_enable_cxx_compiler_flag_if_supported("-pedantic")
_enable_cxx_compiler_flag_if_supported("-Wall")
_enable_cxx_compiler_flag_if_supported("-Wfloat-equal")
_enable_cxx_compiler_flag_if_supported("-Wundef")
_enable_cxx_compiler_flag_if_supported("-Wcast-align")
_enable_cxx_compiler_flag_if_supported("-Wwrite-strings")
_enable_cxx_compiler_flag_if_supported("-Wmissing-declarations")
_enable_cxx_compiler_flag_if_supported("-Wredundant-decls")
_enable_cxx_compiler_flag_if_supported("-Wshadow")
_enable_cxx_compiler_flag_if_supported("-Woverloaded-virtual")




