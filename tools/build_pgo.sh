#!/usr/bin/env bash

set -x

python -c 'import sys;assert sys.platform == "win32"'
is_win=$?

set -e
# Create venv for instrumented build and test
python -m venv build_pgo

if [[ $is_win -eq 0 ]]; then
    source build_pgo/Scripts/activate
else
    source build_pgo/bin/activate
fi

# Build with instrumentation
pip install --prefer-binary -c constraints.txt -r requirements.txt setuptools-rust wheel
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" pip install --prefer-binary -c constraints.txt -e .
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" python setup.py build_rust --release --inplace
pip install -c constraints.txt --prefer-binary -r requirements-dev.txt
# Run profile data generation

QISKIT_PARALLEL=FALSE stestr run --abbreviate

deactivate

${HOME}/.rustup/toolchains/*x86_64*/lib/rustlib/x86_64*/bin/llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
