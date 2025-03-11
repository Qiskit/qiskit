#!/usr/bin/env bash

set -x

work_dir="$1"
out_path="$2"

set -e

# Create venv for instrumented build and test
python -m venv build_pgo

if python -c 'import sys; assert sys.platform == "win32"'; then
    source build_pgo/Scripts/activate
else
    source build_pgo/bin/activate
fi

arch=`uname -m`
# Handle macOS calling the architecture arm64 and rust calling it aarch64
if [[ $arch == "arm64" ]]; then
    arch="aarch64"
fi

# On macOS, manually link the python libraries
if [[ `uname -s` == "Darwin" ]] ; then
    PYLIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    PYNAME=$(find $PYLIB -maxdepth 1 -name "libpython*" | grep -oE "[^/]+$" | grep -oE "python[0-9]+\.[0-9]+" || echo "python")
    LDFLAGS="-L $PYLIB -l $PYNAME"
else
    LDFLAGS=""
fi

# Build with instrumentation
pip install -U -c constraints.txt setuptools-rust wheel setuptools
RUSTFLAGS="-Cprofile-generate=$work_dir $LDFLAGS" pip install --prefer-binary -c constraints.txt -r requirements-dev.txt -e .
RUSTFLAGS="-Cprofile-generate=$work_dir $LDFLAGS" python setup.py build_rust --release --inplace
# Run profile data generation

QISKIT_PARALLEL=FALSE stestr run --abbreviate

python tools/pgo_scripts/test_utility_scale.py

deactivate

${HOME}/.rustup/toolchains/*$arch*/lib/rustlib/$arch*/bin/llvm-profdata merge -o "$out_path" "$work_dir"
