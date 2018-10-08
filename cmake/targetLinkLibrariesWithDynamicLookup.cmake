#.rst:
#
# Public Functions
# ^^^^^^^^^^^^^^^^
#
# The following functions are defined:
#
# .. cmake:command:: target_link_libraries_with_dynamic_lookup
#
# ::
#
#     target_link_libraries_with_dynamic_lookup(<Target> [<Libraries>])
#
#
# Useful to "weakly" link a loadable module. For example, it should be used
# when compiling a loadable module when the symbols should be resolve from
# the run-time environment where the module is loaded, and not a specific
# system library.
#
# Like proper linking, except that the given ``<Libraries>`` are not necessarily
# linked. Instead, the ``<Target>`` is produced in a manner that allows for
# symbols unresolved within it to be resolved at runtime, presumably by the
# given ``<Libraries>``.  If such a target can be produced, the provided
# ``<Libraries>`` are not actually linked.
#
# It links a library to a target such that the symbols are resolved at
# run-time not link-time.
#
# The linker is checked to see if it supports undefined
# symbols when linking a shared library. If it does then the library
# is not linked when specified with this function.
#
# On platforms that do not support weak-linking, this function works just
# like ``target_link_libraries``.
#
# .. note::
#
#     For OSX it uses ``undefined dynamic_lookup``. This is similar to using
#     ``-shared`` on Linux where undefined symbols are ignored.
#
#     For more details, see `blog <http://blog.tim-smith.us/2015/09/python-extension-modules-os-x/>`_
#     from Tim D. Smith.
#
#
# .. cmake:command:: check_dynamic_lookup
#
# Check if the linker requires a command line flag to allow leaving symbols
# unresolved when producing a target of type ``<TargetType>`` that is
# weakly-linked against a dependency of type ``<LibType>``.
#
# ``<TargetType>``
#   can be one of "STATIC", "SHARED", "MODULE", or "EXE".
#
# ``<LibType>``
#   can be one of "STATIC", "SHARED", or "MODULE".
#
# Long signature:
#
# ::
#
#     check_dynamic_lookup(<TargetType>
#                          <LibType>
#                          <ResultVar>
#                          [<LinkFlagsVar>])
#
#
# Short signature:
#
# ::
#
#     check_dynamic_lookup(<ResultVar>) # <TargetType> set to "MODULE"
#                                       # <LibType> set to "SHARED"
#
#
# The result is cached between invocations and recomputed only when the value
# of CMake's linker flag list changes; ``CMAKE_STATIC_LINKER_FLAGS`` if
# ``<TargetType>`` is "STATIC", and ``CMAKE_SHARED_LINKER_FLAGS`` otherwise.
#
#
# Defined variables:
#
# ``<ResultVar>``
#   Whether the current C toolchain supports weak-linking for target binaries of
#   type ``<TargetType>`` that are weakly-linked against a dependency target of
#   type ``<LibType>``.
#
# ``<LinkFlagsVar>``
#   List of flags to add to the linker command to produce a working target
#   binary of type ``<TargetType>`` that is weakly-linked against a dependency
#   target of type ``<LibType>``.
#
# ``HAS_DYNAMIC_LOOKUP_<TargetType>_<LibType>``
#   Cached, global alias for ``<ResultVar>``
#
# ``DYNAMIC_LOOKUP_FLAGS_<TargetType>_<LibType>``
#   Cached, global alias for ``<LinkFlagsVar>``
#
#
# Private Functions
# ^^^^^^^^^^^^^^^^^
#
# The following private functions are defined:
#
# .. warning:: These functions are not part of the scikit-build API. They
#     exist purely as an implementation detail and may change from version
#     to version without notice, or even be removed.
#
#     We mean it.
#
#
# .. cmake:command:: _get_target_type
#
# ::
#
#     _get_target_type(<ResultVar> <Target>)
#
#
# Shorthand for querying an abbreviated version of the target type
# of the given ``<Target>``.
#
# ``<ResultVar>`` is set to:
#
# - "STATIC" for a STATIC_LIBRARY,
# - "SHARED" for a SHARED_LIBRARY,
# - "MODULE" for a MODULE_LIBRARY,
# - and "EXE" for an EXECUTABLE.
#
# Defined variables:
#
# ``<ResultVar>``
#   The abbreviated version of the ``<Target>``'s type.
#
#
# .. cmake:command:: _test_weak_link_project
#
# ::
#
#     _test_weak_link_project(<TargetType>
#                             <LibType>
#                             <ResultVar>
#                             <LinkFlagsVar>)
#
#
# Attempt to compile and run a test project where a target of type
# ``<TargetType>`` is weakly-linked against a dependency of type ``<LibType>``:
#
# - ``<TargetType>`` can be one of "STATIC", "SHARED", "MODULE", or "EXE".
# - ``<LibType>`` can be one of "STATIC", "SHARED", or "MODULE".
#
# Defined variables:
#
# ``<ResultVar>``
#   Whether the current C toolchain can produce a working target binary of type
#   ``<TargetType>`` that is weakly-linked against a dependency target of type
#   ``<LibType>``.
#
# ``<LinkFlagsVar>``
#   List of flags to add to the linker command to produce a working target
#   binary of type ``<TargetType>`` that is weakly-linked against a dependency
#   target of type ``<LibType>``.
#

function(_get_target_type result_var target)
  set(target_type "SHARED_LIBRARY")
  if(TARGET ${target})
    get_property(target_type TARGET ${target} PROPERTY TYPE)
  endif()

  set(result "STATIC")

  if(target_type STREQUAL "STATIC_LIBRARY")
    set(result "STATIC")
  endif()

  if(target_type STREQUAL "SHARED_LIBRARY")
    set(result "SHARED")
  endif()

  if(target_type STREQUAL "MODULE_LIBRARY")
    set(result "MODULE")
  endif()

  if(target_type STREQUAL "EXECUTABLE")
    set(result "EXE")
  endif()

  set(${result_var} ${result} PARENT_SCOPE)
endfunction()


function(_test_weak_link_project
         target_type
         lib_type
         can_weak_link_var
         project_name)

  set(gnu_ld_ignore      "-Wl,--unresolved-symbols=ignore-all")
  set(osx_dynamic_lookup           "-undefined dynamic_lookup")
  set(no_flag                                               "")

  foreach(link_flag_spec gnu_ld_ignore osx_dynamic_lookup no_flag)
    set(link_flag "${${link_flag_spec}}")

    set(test_project_dir "${PROJECT_BINARY_DIR}/CMakeTmp")
    set(test_project_dir "${test_project_dir}/${project_name}")
    set(test_project_dir "${test_project_dir}/${link_flag_spec}")
    set(test_project_dir "${test_project_dir}/${target_type}")
    set(test_project_dir "${test_project_dir}/${lib_type}")

    set(test_project_src_dir "${test_project_dir}/src")
    set(test_project_bin_dir "${test_project_dir}/build")

    file(MAKE_DIRECTORY ${test_project_src_dir})
    file(MAKE_DIRECTORY ${test_project_bin_dir})

    set(mod_type "STATIC")
    set(link_mod_lib TRUE)
    set(link_exe_lib TRUE)
    set(link_exe_mod FALSE)

    if("${target_type}" STREQUAL "EXE")
      set(link_exe_lib FALSE)
      set(link_exe_mod TRUE)
    else()
      set(mod_type "${target_type}")
    endif()

    if("${mod_type}" STREQUAL "MODULE")
      set(link_mod_lib FALSE)
    endif()


    file(WRITE "${test_project_src_dir}/CMakeLists.txt" "
      cmake_minimum_required(VERSION ${CMAKE_VERSION})
      project(${project_name} C)

      include_directories(${test_project_src_dir})

      add_library(number ${lib_type} number.c)
      add_library(counter ${mod_type} counter.c)
    ")

    if("${mod_type}" STREQUAL "MODULE")
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        set_target_properties(counter PROPERTIES PREFIX \"\")
      ")
    endif()

    if(link_mod_lib)
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        target_link_libraries(counter number)
      ")
    elseif(NOT link_flag STREQUAL "")
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        set_target_properties(counter PROPERTIES LINK_FLAGS \"${link_flag}\")
      ")
    endif()

    file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
      add_executable(main main.c)
    ")

    if(link_exe_lib)
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        target_link_libraries(main number)
      ")
    elseif(NOT link_flag STREQUAL "")
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        target_link_libraries(main \"${link_flag}\")
      ")
    endif()

    if(link_exe_mod)
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        target_link_libraries(main counter)
      ")
    else()
      file(APPEND "${test_project_src_dir}/CMakeLists.txt" "
        target_link_libraries(main \"${CMAKE_DL_LIBS}\")
      ")
    endif()

    file(WRITE "${test_project_src_dir}/number.c" "
      #include <number.h>

      static int _number;
      void set_number(int number) { _number = number; }
      int get_number() { return _number; }
    ")

    file(WRITE "${test_project_src_dir}/number.h" "
      #ifndef _NUMBER_H
      #define _NUMBER_H
      extern void set_number(int);
      extern int get_number(void);
      #endif
    ")

    file(WRITE "${test_project_src_dir}/counter.c" "
      #include <number.h>
      int count() {
        int result = get_number();
        set_number(result + 1);
        return result;
      }
    ")

    file(WRITE "${test_project_src_dir}/counter.h" "
      #ifndef _COUNTER_H
      #define _COUNTER_H
      extern int count(void);
      #endif
    ")

    file(WRITE "${test_project_src_dir}/main.c" "
      #include <stdlib.h>
      #include <stdio.h>
      #include <number.h>
    ")

    if(NOT link_exe_mod)
      file(APPEND "${test_project_src_dir}/main.c" "
        #include <dlfcn.h>
      ")
    endif()

    file(APPEND "${test_project_src_dir}/main.c" "
      int my_count() {
        int result = get_number();
        set_number(result + 1);
        return result;
      }

      int main(int argc, char **argv) {
        int result;
    ")

    if(NOT link_exe_mod)
      file(APPEND "${test_project_src_dir}/main.c" "
        void *counter_module;
        int (*count)(void);

        counter_module = dlopen(\"./counter.so\", RTLD_LAZY | RTLD_GLOBAL);
        if(!counter_module) goto error;

        count = dlsym(counter_module, \"count\");
        if(!count) goto error;
      ")
    endif()

    file(APPEND "${test_project_src_dir}/main.c" "
        result = count()    != 0 ? EXIT_FAILURE :
                 my_count() != 1 ? EXIT_FAILURE :
                 my_count() != 2 ? EXIT_FAILURE :
                 count()    != 3 ? EXIT_FAILURE :
                 count()    != 4 ? EXIT_FAILURE :
                 count()    != 5 ? EXIT_FAILURE :
                 my_count() != 6 ? EXIT_FAILURE : EXIT_SUCCESS;
    ")

    if(NOT link_exe_mod)
      file(APPEND "${test_project_src_dir}/main.c" "
        goto done;
        error:
          fprintf(stderr, \"Error occured:\\n    %s\\n\", dlerror());
          result = 1;

        done:
          if(counter_module) dlclose(counter_module);
      ")
    endif()

    file(APPEND "${test_project_src_dir}/main.c" "
          return result;
      }
    ")

    set(_rpath_arg)
    if(APPLE AND ${CMAKE_VERSION} VERSION_GREATER 2.8.11)
      set(_rpath_arg "-DCMAKE_MACOSX_RPATH='${CMAKE_MACOSX_RPATH}'")
    endif()

    try_compile(project_compiles
                "${test_project_bin_dir}"
                "${test_project_src_dir}"
                "${project_name}"
                CMAKE_FLAGS
                  "-DCMAKE_SHARED_LINKER_FLAGS='${CMAKE_SHARED_LINKER_FLAGS}'"
                  "-DCMAKE_ENABLE_EXPORTS=ON"
                  ${_rpath_arg}
                OUTPUT_VARIABLE compile_output)

    set(project_works 1)
    set(run_output)

    if(project_compiles)
      execute_process(COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
                              "${test_project_bin_dir}/main"
                      WORKING_DIRECTORY "${test_project_bin_dir}"
                      RESULT_VARIABLE project_works
                      OUTPUT_VARIABLE run_output
                      ERROR_VARIABLE run_output)
    endif()

    set(test_description
        "Weak Link ${target_type} -> ${lib_type} (${link_flag_spec})")

    if(project_works EQUAL 0)
      set(project_works TRUE)
      message(STATUS "Performing Test ${test_description} - Success")
    else()
      set(project_works FALSE)
      message(STATUS "Performing Test ${test_description} - Failed")
      file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log
           "Performing Test ${test_description} failed with the "
           "following output:\n"
           "BUILD\n-----\n${compile_output}\nRUN\n---\n${run_output}\n")
    endif()

    set(${can_weak_link_var} ${project_works} PARENT_SCOPE)
    if(project_works)
      set(${project_name} ${link_flag} PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

function(check_dynamic_lookup)
  # Two signatures are supported:

  if(ARGC EQUAL "1")
    #
    # check_dynamic_lookup(<ResultVar>)
    #
    set(target_type "MODULE")
    set(lib_type "SHARED")
    set(has_dynamic_lookup_var "${ARGV0}")
    set(link_flags_var "unused")

  elseif(ARGC GREATER "2")
    #
    # check_dynamic_lookup(<TargetType>
    #                      <LibType>
    #                      <ResultVar>
    #                      [<LinkFlagsVar>])
    #
    set(target_type "${ARGV0}")
    set(lib_type "${ARGV1}")
    set(has_dynamic_lookup_var "${ARGV2}")
    if(ARGC EQUAL "3")
      set(link_flags_var "unused")
    else()
      set(link_flags_var "${ARGV3}")
    endif()
  else()
    message(FATAL_ERROR "missing arguments")
  endif()

  _check_dynamic_lookup(
    ${target_type}
    ${lib_type}
    ${has_dynamic_lookup_var}
    ${link_flags_var}
    )
  set(${has_dynamic_lookup_var} ${${has_dynamic_lookup_var}} PARENT_SCOPE)
  if(NOT "x${link_flags_var}x" STREQUAL "xunusedx")
    set(${link_flags_var} ${${link_flags_var}} PARENT_SCOPE)
  endif()
endfunction()

function(_check_dynamic_lookup
         target_type
         lib_type
         has_dynamic_lookup_var
         link_flags_var
         )

  # hash the CMAKE_FLAGS passed and check cache to know if we need to rerun
  if("${target_type}" STREQUAL "STATIC")
    string(MD5 cmake_flags_hash "${CMAKE_STATIC_LINKER_FLAGS}")
  else()
    string(MD5 cmake_flags_hash "${CMAKE_SHARED_LINKER_FLAGS}")
  endif()

  set(cache_var "HAS_DYNAMIC_LOOKUP_${target_type}_${lib_type}")
  set(cache_hash_var "HAS_DYNAMIC_LOOKUP_${target_type}_${lib_type}_hash")
  set(result_var "DYNAMIC_LOOKUP_FLAGS_${target_type}_${lib_type}")

  if(     NOT DEFINED ${cache_hash_var}
       OR NOT "${${cache_hash_var}}" STREQUAL "${cmake_flags_hash}")
    unset(${cache_var} CACHE)
  endif()

  if(NOT DEFINED ${cache_var})
    set(skip_test FALSE)

   if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
      set(skip_test TRUE)
    endif()

    if(skip_test)
      set(has_dynamic_lookup FALSE)
      set(link_flags)
    else()
      _test_weak_link_project(${target_type}
                              ${lib_type}
                              has_dynamic_lookup
                              link_flags)
    endif()

    set(caveat " (when linking ${target_type} against ${lib_type})")

    set(${cache_var} "${has_dynamic_lookup}"
        CACHE BOOL
        "linker supports dynamic lookup for undefined symbols${caveat}")
    mark_as_advanced(${cache_var})

    set(${result_var} "${link_flags}"
        CACHE STRING
        "linker flags for dynamic lookup${caveat}")
    mark_as_advanced(${result_var})

    set(${cache_hash_var} "${cmake_flags_hash}"
        CACHE INTERNAL "hashed flags for ${cache_var} check")
  endif()

  set(${has_dynamic_lookup_var} "${${cache_var}}" PARENT_SCOPE)
  set(${link_flags_var} "${${result_var}}" PARENT_SCOPE)
endfunction()

function(target_link_libraries_with_dynamic_lookup target)
  _get_target_type(target_type ${target})

  set(link_props)
  set(link_items)
  set(link_libs)

  foreach(lib ${ARGN})
    _get_target_type(lib_type ${lib})
    check_dynamic_lookup(${target_type}
                         ${lib_type}
                         has_dynamic_lookup
                         dynamic_lookup_flags)

    if(has_dynamic_lookup)
      if(dynamic_lookup_flags)
        if("${target_type}" STREQUAL "EXE")
          list(APPEND link_items "${dynamic_lookup_flags}")
        else()
          list(APPEND link_props "${dynamic_lookup_flags}")
        endif()
      endif()
    elseif(${lib} MATCHES "(debug|optimized|general)")
      # See gh-255
    else()
      list(APPEND link_libs "${lib}")
    endif()
  endforeach()

  if(link_props)
    list(REMOVE_DUPLICATES link_props)
  endif()

  if(link_items)
    list(REMOVE_DUPLICATES link_items)
  endif()

  if(link_libs)
    list(REMOVE_DUPLICATES link_libs)
  endif()

  if(link_props)
    set_target_properties(${target}
                          PROPERTIES LINK_FLAGS "${link_props}")
  endif()

  set(links "${link_items}" "${link_libs}")
  if(links)
    target_link_libraries(${target} "${links}")
  endif()
endfunction()

