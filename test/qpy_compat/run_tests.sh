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

# Set fixed hash seed to ensure set orders are identical between saving and
# loading.
export PYTHONHASHSEED=$(python -S -c "import random; print(random.randint(1, 4294967295))")
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"

qiskit_venv="$(pwd -P)/qiskit_venv"
qiskit_python="$qiskit_venv/bin/python"
python -m venv "$qiskit_venv"
# `packaging` is needed for the `get_versions.py` script.
#  symengine is pinned to 0.13 to explicitly test the migration path reusing the venv
"$qiskit_venv/bin/pip" install -c "$our_dir/../../constraints.txt" "$our_dir/../.." packaging "symengine~=0.13"

"$qiskit_python" "$our_dir/get_versions.py" | parallel --colsep=" " bash "$our_dir/process_version.sh" -p "$qiskit_python"

# Test dev compatibility
dev_version="$("$qiskit_python" -c 'import qiskit; print(qiskit.__version__)')"
"$qiskit_python" "$our_dir/test_qpy.py" generate --version="$dev_version"
"$qiskit_python" "$our_dir/test_qpy.py" load --version="$dev_version"

# Test dev compatibility with different symengine versions across 0.11 and 0.13
#
# NOTE: When symengine >= 0.14.0 is released we will need to modify this to build an explicit
# symengine 0.13.0 venv instead of reusing $qiskit_venv.
#
symengine_11_venv="$(pwd -P)/qiskit_symengine_11_venv"
symengine_11_python="$symengine_11_venv/bin/python"
python -m venv "$symengine_11_venv"
"$symengine_11_venv/bin/pip" install -c "$our_dir/../../constraints.txt" "$our_dir/../.." "symengine==0.11.0"
# Load symengine 0.13.0 generated payload with symengine 0.11
"$symengine_11_python" "$our_dir/test_qpy.py" load --version="$dev_version"
# Load symengine 0.11.0 generated payload with symengine 0.13.0
mkdir symengine_11_qpy_files
pushd symengine_11_qpy_files
"$symengine_11_python" "$our_dir/test_qpy.py" generate --version="$dev_version"
"$qiskit_python" "$our_dir/test_qpy.py" load --version="$dev_version"
popd
