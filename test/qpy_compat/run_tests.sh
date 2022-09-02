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
set -x

# Set fixed hash seed to ensure set orders are identical between saving and
# loading.
export PYTHONHASHSEED=$(python -S -c "import random; print(random.randint(1, 4294967295))")
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

python -m venv qiskit_venv
qiskit_venv/bin/pip install ../..
python_dev="qiskit_venv/bin/python"

for version in $(git tag --sort=-creatordate) ; do
    parts=( ${version//./ } )
    if [[ ${parts[1]} -lt 18 ]] ; then
        break
    fi
    python_stable="stable_venv/$version/bin/python"
    # Build venv if necessary.
    if [[ ! -x "${python_stable}" ]] || ! ${python_stable} -c 'import qiskit'; then
        echo "Building venv for qiskit-terra $version"
        # Remove corrupt venv.
        rm -rf stable_venv/$version
        python -m venv stable_venv/$version
        ./stable_venv/$version/bin/pip install "qiskit-terra==$version"
    fi
    echo "Generating qpy files with qiskit-terra $version"
    "$python_stable" test_qpy.py generate --version=$version
    echo "Loading qpy files from $version with dev qiskit-terra"
    "$python_dev" test_qpy.py load --version=$version
    rm *qpy
done

# Test dev compatibility
dev_version=$("$python_dev" -c 'import qiskit;print(qiskit.__version__)')
"$python_dev" test_qpy.py generate --version=$dev_version
"$python_dev" test_qpy.py load --version=$dev_version
