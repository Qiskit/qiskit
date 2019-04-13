# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,invalid-sequence-index,unused-argument
# pylint: disable=unsupported-assignment-operation

"""
Methods to assist with compiling tasks.
"""
import warnings

from qiskit.quantum_info import Unitary
from qiskit.quantum_info import synthesis


def euler_angles_1q(unitary_matrix):
    """Deprecated after 0.8
    """
    warnings.warn("euler_angles_1q function is now accessible under "
                  "qiskit.quantum_info.synthesis", DeprecationWarning)
    return synthesis.euler_angles_1q(unitary_matrix)


def two_qubit_kak(unitary_matrix, verify_gate_sequence=False):
    """Deprecated after 0.8
    """
    warnings.warn("two_qubit_kak function is now accessible under "
                  "qiskit.quantum_info.synthesis", DeprecationWarning)
    return synthesis.two_qubit_kak(Unitary(unitary_matrix))
