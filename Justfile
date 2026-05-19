# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

set shell := ['sh', '-e', '-u', '-c']
set windows-shell := ['powershell.exe', '-NoProfile', '-c', '$ErrorActionPreference = "Stop";']

mkdir_p := if os() == 'windows' { 'mkdir -Fo' } else { 'mkdir -p' }
rm_rf := if os() == 'windows' { 'Write-Output ' } else { 'rm -rf' }
end_rm_rf := if os() == 'windows' { '| rm -Rec -Fo -ErrorAction SilentlyContinue ; exit 0' } else { '' }

mimalloc := if env('QISKIT_BUILD_WITH_MIMALLOC', '') == '1' {
	'--features=mimalloc'
} else {
	''
}

default: style lint-incr test

# Dependencies need to be installed on the Anaconda virtual environment.
[unix]
env:
	if ! conda info --envs | grep -iq 'qiskitenv'; then \
		conda create -y -n Qiskitenv python=3; \
	fi
	conda run -n Qiskitenv pip install -r requirements.txt

[windows]
env:
	if (-not ({ conda info --envs } -match "(?i)qiskitenv")) { \
		conda create -y -n Qiskitenv python=3 \
	}
	conda run -n Qiskitenv pip install -r requirements.txt

lint:
	ruff check qiskit test tools setup.py
	python3 tools/verify_headers.py qiskit test tools
	python3 tools/find_optional_imports.py
	python3 tools/find_stray_release_notes.py
	python3 tools/verify_images.py

lint-incr:
	ruff check qiskit test tools setup.py
	python3 tools/verify_headers.py qiskit test tools
	python3 tools/find_optional_imports.py
	python3 tools/verify_images.py

ruff:
	ruff qiskit test tools setup.py

style:
	black --check qiskit test tools setup.py docs/conf.py

black:
	black qiskit test tools setup.py docs/conf.py

# Use the -s (starting directory) flag for "unittest discover" is necessary,
# otherwise the QuantumCircuit header will be modified during the discovery.
test:
	@echo '================================================'
	@echo 'Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci'
	@echo '================================================'
	python3 -m unittest discover -s test/python -t . -v
	@echo '================================================'
	@echo 'Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci'
	@echo '================================================'

# Use pytest to run tests
pytest:
	pytest test/python

# Use pytest to run randomized tests
pytest_randomized:
	pytest test/randomized

test_ci:
	QISKIT_TEST_CAPTURE_STREAMS=1 stestr run

test_randomized:
	python3 -m unittest discover -s test/randomized -t . -v

coverage:
	coverage run --source qiskit -m unittest discover -s test/python -q
	coverage report

coverage_erase:
	coverage erase

clean: coverage_erase

# ==============================================================================
# Variables that can be set/modified to modify the C builds.
# ==============================================================================

# Output directories.
c_dir_out := 'dist/c'
c_dir_out_lib := c_dir_out + '/lib'
c_dir_out_include := c_dir_out + '/include'
c_dir_test := 'test/c'
c_dir_test_build := 'test/c/build'

# Input directories
c_dir_cargo_target := 'target'
c_dir_crates := 'crates'

# ==============================================================================
# Variables that we derive from the settings above.
# ==============================================================================

# `else` branch here is just a guess
c_lib_cargo_filename := if os() == 'windows' {
	'qiskit_cext.dll'
} else if os() == "macos" {
	'libqiskit_cext.dylib'
} else {
	'libqiskit_cext.so'
}

c_libqiskit_out := c_dir_out_lib + '/' + replace(c_lib_cargo_filename, '_cext', '')

# ==============================================================================
# Recipes for the C components.
# ==============================================================================

# Run clang-format (does not apply any changes)
cformat:
	bash tools/run_clang_format.sh
# Apply clang-format changes
fix_cformat:
	bash tools/run_clang_format.sh apply

# Abstraction over calling Cargo to build the C extension in "standalone" C
# mode.  This _also_ builds the C header file as a side-effect into
# `target/qiskit.h`.  Recipes that use this as a prerequisite should ensure they
# set `C_LIB_CARGO_FLAGS` to choose the build profile.  The `C_LIB_RUSTC_FLAGS`
# variable can also be set to add additional logic (like coverage instructions).
#
# Typically, downstream recipes should depend on `build-clib-release` or `build-clib-dev`
# instead.

env_c_lib_cargo_flags := env('C_LIB_CARGO_FLAGS', '')
env_c_lib_rustc_flags := env('C_LIB_RUSTC_FLAGS', '')

build-clib c_lib_cargo_flags=env_c_lib_cargo_flags:
	cargo rustc -p qiskit-cext {{mimalloc}} --crate-type cdylib {{c_lib_cargo_flags}} -- {{env_c_lib_rustc_flags}}
build-clib-release: (build-clib '--release')
build-clib-dev: (build-clib '--profile dev')

# Catch-all directory-creation rule.
_create_c_dir_out_lib:
	{{mkdir_p}} '{{c_dir_out_lib}}'

cheader:
	cargo run -p qiskit-bindgen-cli -- install -c '{{c_dir_crates}}/cext' -o '{{c_dir_out_include}}'
# `clib` and `clib-dev` are conflicting rules - they both attempt to "install" the
# shared library into the output `lib` directory, but they differ between release
# and dev mode.
clib: build-clib-release _create_c_dir_out_lib
	cp '{{c_dir_cargo_target}}/release/{{c_lib_cargo_filename}}' '{{c_libqiskit_out}}'
clib-dev: build-clib-dev _create_c_dir_out_lib
	cp '{{c_dir_cargo_target}}/debug/{{c_lib_cargo_filename}}' '{{c_libqiskit_out}}'
c: cheader clib

# Use ctest to run C API tests.
ctest $C_LIB_RUSTC_FLAGS='': cheader build-clib-dev
	# `-S` specifies the source (including the `CMakeLists.txt` file, `-B` is where
	# to put the build files, including the generated CMake stuff.  See the
	# `CMakeLists.txt` file for the build variables.
	cmake '-S{{c_dir_test}}' '-B{{c_dir_test_build}}' \
		'-DCARGO_LIB_DIR={{absolute_path(c_dir_cargo_target)}}/debug' \
		'-DQISKIT_INCLUDE_PATH={{absolute_path(c_dir_out_include)}}'
	# Actually build the test.
	cmake --build '{{c_dir_test_build}}'
	# -V ensures we always produce a logging output to indicate the subtests
	# -C Debug is needed for windows to work, if you don't specify Debug (or
	# Release) explicitly ctest doesn't run on windows
	ctest -V -C Debug --test-dir '{{c_dir_test_build}}'

ccoverage: (ctest "-Cinstrument-coverage")

cclean:
	{{rm_rf}} '{{c_dir_out}}' '{{c_dir_test_build}}' {{end_rm_rf}}
	cargo clean --package qiskit-cext
