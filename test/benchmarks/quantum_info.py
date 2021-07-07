# -*- coding: utf-8 -*

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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

from qiskit.quantum_info import random_clifford, Clifford, decompose_clifford
from qiskit.quantum_info import random_cnotdihedral, CNOTDihedral
import numpy as np


class RandomCliffordBench:
    params = ['1,3000', '2,2500', '3,2000', '4,1500', '5,1000', '6,700']
    param_names = ['nqubits,length']

    def time_random_clifford(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        for _ in range(length):
            random_clifford(nqubits)


class CliffordComposeBench:
    params = ['1,7000', '2,5000', '3,5000', '4,2500', '5,2000']
    param_names = ['nqubits,length']

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        self.random_clifford = \
            [random_clifford(nqubits) for _ in range(length)]

    def time_compose(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        clifford = Clifford(np.eye(2 * nqubits))
        for i in range(length):
            clifford.compose(self.random_clifford[i])


class CliffordDecomposeBench:
    params = ['1,1000', '2,500', '3,100', '4,50', '5,10']
    param_names = ['nqubits,length']

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        self.random_clifford = \
            [random_clifford(nqubits) for _ in range(length)]

    def time_decompose(self, nqubits_length):
        length = int(nqubits_length.split(',')[1])
        for i in range(length):
            decompose_clifford(self.random_clifford[i])


class RandomCnotDihedralBench:
    params = ['1,2000', '2,1500', '3,1200', '4,1000', '5,800', '6,700']
    param_names = ['nqubits,length']

    def time_random_cnotdihedral(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        for _ in range(length):
            random_cnotdihedral(nqubits)


class CnotDihedralComposeBench:
    params = ['1,1500', '2,400', '3,100', '4,40', '5,10']
    param_names = ['nqubits,length']

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        self.random_cnotdihedral = \
            [random_cnotdihedral(nqubits) for _ in range(length)]

    def time_compose(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(','))
        cxdihedral = CNOTDihedral(num_qubits=nqubits)
        for i in range(length):
            cxdihedral.compose(self.random_cnotdihedral[i])
