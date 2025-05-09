# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

ifneq ($(OS), Windows_NT)
	OS := $(shell uname -s)
endif

.PHONY: default ruff env lint lint-incr style black test test_randomized pytest pytest_randomized test_ci coverage coverage_erase clean cheader clib ctest cformat fix_cformat cclean

default: ruff style lint-incr test ;

# Dependencies need to be installed on the Anaconda virtual environment.
env:
	if test $(findstring qiskitenv, $(shell conda info --envs | tr '[:upper:]' '[:lower:]')); then \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	else \
		conda create -y -n Qiskitenv python=3; \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	fi;

# Ignoring generated ones with .py extension.
lint:
	pylint -rn qiskit test tools
	tools/verify_headers.py qiskit test tools
	tools/find_optional_imports.py
	tools/find_stray_release_notes.py
	tools/verify_images.py

# Only pylint on files that have changed from origin/main. Also parallelize (disables cyclic-import check)
lint-incr:
	-git fetch -q https://github.com/Qiskit/qiskit-terra.git :lint_incr_latest
	tools/pylint_incr.py -j4 -rn -sn --paths :/qiskit/*.py :/test/*.py :/tools/*.py
	tools/verify_headers.py qiskit test tools
	tools/find_optional_imports.py
	tools/verify_images.py

ruff:
	ruff qiskit test tools setup.py

style:
	black --check qiskit test tools setup.py

black:
	black qiskit test tools setup.py

# Use the -s (starting directory) flag for "unittest discover" is necessary,
# otherwise the QuantumCircuit header will be modified during the discovery.
test:
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================
	python3 -m unittest discover -s test/python -t . -v
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================

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
	coverage3 run --source qiskit -m unittest discover -s test/python -q
	coverage3 report

coverage_erase:
	coverage erase

clean: coverage_erase ;

C_DIR_OUT = dist/c
C_DIR_LIB = $(C_DIR_OUT)/lib
C_DIR_INCLUDE = $(C_DIR_OUT)/include
C_DIR_TEST_BUILD = test/c/build
# Whether this is target/debug or target/release depends on the flags in the
# `cheader` recipe.  For now, they're just hardcoded.
C_CARGO_TARGET_DIR = target/release
ifeq ($(OS), Windows_NT)
	C_LIB_CARGO_FILENAME=qiskit_cext.dll
else ifeq ($(shell uname), Darwin)
	C_LIB_CARGO_FILENAME=libqiskit_cext.dylib
else
	# ... probably.
	C_LIB_CARGO_FILENAME=libqiskit_cext.so
endif
C_LIB_CARGO_PATH=$(C_CARGO_TARGET_DIR)/$(C_LIB_CARGO_FILENAME)

C_QISKIT_H=$(C_DIR_INCLUDE)/qiskit.h
C_LIBQISKIT=$(C_DIR_LIB)/$(subst _cext,,$(C_LIB_CARGO_FILENAME))

# Run clang-format (does not apply any changes)
cformat:
	bash tools/run_clang_format.sh

# Apply clang-format changes
fix_cformat:
	bash tools/run_clang_format.sh apply

# The library file is managed by a different build tool - pretend it's always dirty.
.PHONY: $(C_LIB_CARGO_PATH)
$(C_LIB_CARGO_PATH):
	cargo rustc --release --crate-type cdylib -p qiskit-cext

$(C_DIR_LIB):
	mkdir -p $(C_DIR_LIB)

$(C_DIR_INCLUDE):
	mkdir -p $(C_DIR_INCLUDE)

$(C_LIBQISKIT): $(C_DIR_LIB)  $(C_LIB_CARGO_PATH)
	cp $(C_LIB_CARGO_PATH) $(C_DIR_LIB)/$(subst _cext,,$(C_LIB_CARGO_FILENAME))

$(C_QISKIT_H): $(C_DIR_INCLUDE) $(C_LIB_CARGO_PATH)
	cp target/qiskit.h $(C_DIR_INCLUDE)/qiskit.h

.PHONY: c cheader
cheader: $(C_QISKIT_H)
c: $(C_LIBQISKIT) $(C_QISKIT_H)

# Use ctest to run C API tests
ctest: $(C_DIR_INCLUDE)
	cargo rustc --crate-type cdylib -p qiskit-cext
	cp target/qiskit.h $(C_DIR_INCLUDE)/qiskit.h

	# -S specifically specifies the source path to be the current folder
	# -B specifically specifies the build path to be inside test/c/build
	cmake -S. -B$(C_DIR_TEST_BUILD)
	cmake --build $(C_DIR_TEST_BUILD)
	# -V ensures we always produce a logging output to indicate the subtests
	# -C Debug is needed for windows to work, if you don't specify Debug (or
	#  release) explicitly ctest doesn't run on windows
	ctest -V -C Debug --test-dir $(C_DIR_TEST_BUILD)

cclean:
	rm -rf $(C_DIR_OUT) $(C_DIR_TEST_BUILD)
	rm -f target/qiskit.h
	cargo clean
