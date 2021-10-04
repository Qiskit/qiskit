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

python -m venv qiskit_venv
qiskit_venv/bin/pip install ../..

for version in $(git tag --sort=-creatordate) ; do
    parts=( ${version//./ } )
    if [[ ${parts[1]} -lt 18 ]] ; then
        break
    fi
    echo "Building venv for qiskit-terra $version"
    python -m venv $version
    ./$version/bin/pip install "qiskit-terra==$version"
    echo "Generating qpy files with qiskit-terra $version"
    ./$version/bin/python test_qpy.py generate --version=$version
    echo "Loading qpy files from $version with dev qiskit-terra"
    qiskit_venv/bin/python test_qpy.py load --version=$version
    rm *qpy
done

# Test dev compatibility
dev_version=`qiskit_venv/bin/python -c 'import qiskit;print(qiskit.__version__)'`
qiskit_venv/bin/python test_qpy.py generate --version=$dev_version
qiskit_venv/bin/python test_qpy.py load --version=$dev_version
