#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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

version=$1
parts=( ${version//./ } )
if [[ ${parts[1]} -lt 18 ]] ; then
    exit 0
fi
echo "Building venv for qiskit-terra $version"
if [[ ! -d qpy_$version ]] ; then
    python -m venv $version
    ./$version/bin/pip install "qiskit-terra==$version"
    mkdir qpy_$version
    pushd qpy_$version
    echo "Generating qpy files with qiskit-terra $version"
    ../$version/bin/python ../test_qpy.py generate --version=$version
else
    echo "Using cached QPY files for $version"
    pushd qpy_$version
fi
echo "Loading qpy files from $version with dev qiskit-terra"
../qiskit_venv/bin/python ../test_qpy.py load --version=$version
popd
