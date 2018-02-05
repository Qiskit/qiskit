# Creates a TARGET_NAME target for creating Pypi distributable package
# Args:
#    TARGET_NAME    The name of the target to be invoked in the Makefile
#    PACKAGE_TYPE   The type of distributable package we want to build. We have
#                   these options:
#						"sdist" => Source distribution package
#						"bdist_wheel" => Platform specific distribution package
#						"both" => Creates both sdist and bdist_wheel
function(add_pypi_package_target TARGET_NAME PACKAGE_TYPE)
	# Create Python distrubution package
	find_program(PYTHON "python")
	if (NOT PYTHON)
		message(FATAL_ERROR "We can't find Python in your system. Please, install it and try again...")
	endif()
	set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
	set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
	set(EXECUTABLE_FILE_EXTENSION "")

	message("QISKIT_VERSION = ${QISKIT_VERSION}")
	configure_file(${SETUP_PY_IN} ${SETUP_PY})
	# For ' make clean' target
	set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${SETUP_PY})

	if(PACKAGE_TYPE STREQUAL "both")
		set(PIP_PACKAGE_SOURCE_DIST sdist --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
		set(PIP_PACKAGE_PLATFORM_WHEELS bdist_wheel --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	elseif(PACKAGE_TYPE STREQUAL "sdist")
		set(PIP_PACKAGE_SOURCE_DIST sdist --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	elseif(PACKAGE_TYPE STREQUAL "bdist_wheel")
		set(PIP_PACKAGE_PLATFORM_WHEELS bdist_wheel --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	endif()

	# For ' make clean' target
	set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
		${CMAKE_CURRENT_BINARY_DIR}/dist)

	if(MINGW)
		set(EXECUTABLE_FILE_EXTENSION ".exe")
	endif()

	# The main package target, all othe targets will depend on it
	add_custom_target(${TARGET_NAME})

	# For source distributions, we don't want any binary in the final package
	if(PIP_PACKAGE_SOURCE_DIST)
		set(TARGET_NAME_SDIST ${TARGET_NAME}_SDIST)
		add_custom_target(${TARGET_NAME_SDIST})
		add_custom_command(TARGET ${TARGET_NAME_SDIST}
			COMMAND ${PYTHON} ${SETUP_PY} ${PIP_PACKAGE_SOURCE_DIST}
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	endif()

	if(PIP_PACKAGE_PLATFORM_WHEELS)
		set(COPY_QISKIT_SIM_TARGET ${TARGET_NAME}_COPY_QISKIT_SIMULATOR)
		# We create a target which will depend on TARGET_NAME_WHEELS for
		# copying all the binaries to their final locations
		add_custom_target(${COPY_QISKIT_SIM_TARGET})
    	add_custom_command(TARGET ${COPY_QISKIT_SIM_TARGET}
			COMMAND ${CMAKE_COMMAND} -E copy
				${QISKIT_SIMULATOR_OUTPUT_DIR}/qiskit_simulator${EXECUTABLE_FILE_EXTENSION}
				${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends)
		# For ' make clean' target
		set_property(DIRECTORY APPEND PROPERTY
			ADDITIONAL_MAKE_CLEAN_FILES
				${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends/qiskit_simulator${EXECUTABLE_FILE_EXTENSION})
		# For Windows, we need to copy external .dll dependencies too
		if(MINGW)
			foreach(dll_file ${QISKIT_SIMULATOR_THIRD_PARTY_DLLS})
				add_custom_command(TARGET ${COPY_QISKIT_SIM_TARGET}
					COMMAND ${CMAKE_COMMAND} -E copy
						${dll_file}
						${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends)
				# For 'make clean' target
				get_filename_component(FINAL_FILE ${dll_file} NAME)
				set_property(DIRECTORY APPEND PROPERTY
					ADDITIONAL_MAKE_CLEAN_FILES
						${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends/${FINAL_FILE})
			endforeach()
		endif()

		set(TARGET_NAME_WHEELS ${TARGET_NAME}_WHEELS)
		add_custom_target(${TARGET_NAME_WHEELS})
		add_custom_command(TARGET ${TARGET_NAME_WHEELS}
			COMMAND ${PYTHON} ${SETUP_PY} ${PIP_PACKAGE_PLATFORM_WHEELS}
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	endif()

	# Create our dependency graph
	if(PIP_PACKAGE_PLATFORM_WHEELS)
		add_dependencies(${TARGET_NAME} ${TARGET_NAME_WHEELS})
		add_dependencies(${TARGET_NAME_WHEELS} ${COPY_QISKIT_SIM_TARGET})
		add_dependencies(${COPY_QISKIT_SIM_TARGET} qiskit_simulator)
		# If we have to build the source distribution as well, then we
		# need to build and package source distribution package first, and the
		# way to express this is by depending on the source distribution target.
		# Otherwise the binaries of the wheel package will be added to the
		# source distribution too, and we don't want that.
		if(PIP_PACKAGE_SOURCE_DIST)
			add_dependencies(${COPY_QISKIT_SIM_TARGET} ${TARGET_NAME_SDIST})
		endif()
	endif()

	if(PIP_PACKAGE_SOURCE_DIST)
		# if we have to build wheels package too, we already have a depdency
		# with TARGET_NAME, but if we haven't, we need to depend on TARGET_NAME
		if(NOT PIP_PACKAGE_PLATFORM_WHEELS)
			add_dependencies(${TARGET_NAME} ${TARGET_NAME_SDIST})
		endif()
		add_dependencies(${TARGET_NAME_SDIST} qiskit_simulator)
	endif()
endfunction()

