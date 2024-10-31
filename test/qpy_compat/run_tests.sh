#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
set -e
set -o pipefail
set -x
shopt -s nullglob

# Set fixed hash seed to ensure set orders are identical between saving and
# loading.
export PYTHONHASHSEED=$(python -S -c "import random; print(random.randint(1, 4294967295))")
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
repo_root="$(realpath -- "$our_dir/../..")"

# First, prepare a wheel file for the dev version.  We install several venvs with this, and while
# cargo will cache some rust artefacts, it still has to re-link each time, so the wheel build takes
# a little while.
wheel_dir="$(pwd -P)/wheels"
python -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$repo_root"
all_wheels=("$wheel_dir"/*.whl)
qiskit_dev_wheel="${all_wheels[0]}"

# Now set up a "base" development-version environment, which we'll use for most of the backwards
# compatibility checks.
qiskit_venv="$(pwd -P)/venvs/dev"
qiskit_python="$qiskit_venv/bin/python"
python -m venv "$qiskit_venv"

# `packaging` is needed for the `get_versions.py` script.
"$qiskit_venv/bin/pip" install -c "$repo_root/constraints.txt" "$qiskit_dev_wheel" packaging

# Run all of the tests of cross-Qiskit-version compatibility.
"$qiskit_python" "$our_dir/get_versions.py" | parallel --colsep=" " bash "$our_dir/process_version.sh" -p "$qiskit_python"

# Test dev compatibility with itself.
dev_version="$("$qiskit_python" -c 'import qiskit; print(qiskit.__version__)')"
mkdir -p "dev-files/base"
pushd "dev-files/base"
"$qiskit_python" "$our_dir/test_qpy.py" generate --version="$dev_version"
"$qiskit_python" "$our_dir/test_qpy.py" load --version="$dev_version"
popd


# Test dev compatibility with all supported combinations of symengine between generator and loader.
# This will likely duplicate the base dev-compatibility test, but the tests are fairly fast, and
# it's better safe than sorry with the serialisation tests.

symengine_versions=(
    '>=0.11,<0.12'
    '>=0.13,<0.14'
)
symengine_venv_prefix="$(pwd -P)/venvs/dev-symengine-"
symengine_files_prefix="$(pwd -P)/dev-files/symengine-"

# Create the venvs and QPY files for each symengine version.
for i in "${!symengine_versions[@]}"; do
    specifier="${symengine_versions[$i]}"
    symengine_venv="$symengine_venv_prefix$i"
    files_dir="$symengine_files_prefix$i"
    python -m venv "$symengine_venv"
    "$symengine_venv/bin/pip" install -c "$repo_root/constraints.txt" "$qiskit_dev_wheel" "symengine$specifier"
    mkdir -p "$files_dir"
    pushd "$files_dir"
    "$symengine_venv/bin/python" -c 'import symengine; print(symengine.__version__)' > "SYMENGINE_VERSION"
    "$symengine_venv/bin/python" "$our_dir/test_qpy.py" generate --version="$dev_version"
    popd
done

# For each symengine version, try loading the QPY files from every other symengine version.
for loader_num in "${!symengine_versions[@]}"; do
    loader_venv="$symengine_venv_prefix$loader_num"
    loader_version="$(< "$symengine_files_prefix$loader_num/SYMENGINE_VERSION")"
    for generator_num in "${!symengine_versions[@]}"; do
        generator_files="$symengine_files_prefix$generator_num"
        generator_version="$(< "$generator_files/SYMENGINE_VERSION")"
        echo "Using symengine==$loader_version to load files generated with symengine==$generator_version"
        pushd "$generator_files"
        "$loader_venv/bin/python" "$our_dir/test_qpy.py" load --version="$dev_version"
        popd
    done
done
