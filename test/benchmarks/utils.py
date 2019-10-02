# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Benchmark utility functions."""

import numpy as np
from qiskit.quantum_info.random import random_unitary
from qiskit import QuantumCircuit


def build_qv_model_circuit(width, depth, seed=None):
    """
    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.
    """
    np.random.seed(seed)
    circuit = QuantumCircuit(width)
    # For each layer
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = np.random.permutation(width)
        # For each pair p in Pj, generate Haar random SU(4)
        for k in range(int(np.floor(width/2))):
            U = random_unitary(4)
            pair = int(perm[2*k]), int(perm[2*k+1])
            circuit.append(U, [pair[0], pair[1]])
    return circuit
