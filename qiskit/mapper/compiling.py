# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,invalid-sequence-index,unused-argument
# pylint: disable=unsupported-assignment-operation

"""
Methods to assist with compiling tasks.
"""

import warnings

from qiskit.exceptions import QiskitError

warnings.warn("The functionality previously under qiskit.mapper.compiling "
              "is now under qiskit.quantum_info.synthesis with a new API")


def euler_angles_1q(unitary_matrix):
    """Moved and API changed after 0.8
    """
    raise QiskitError("euler_angles_1q functionality is now accessible in "
                      "qiskit.quantum_info.synthesis")


def two_qubit_kak(unitary_matrix, verify_gate_sequence=False):
    """Moved and API changed after 0.8
    """
    raise QiskitError("two_qubit_kak() functionality is now available in "
                      "qiskit.quantum_info.synthesis.two_qubit_cnot_decompose()")
