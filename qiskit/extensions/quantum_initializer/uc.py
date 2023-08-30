# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The structure of the code is based on Emanuel Malvetti's semester thesis at
# ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.

"""
Uniformly controlled gates (also called multiplexed gates).

These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state |i> (in the computational basis),
a single-qubit unitary U_i is applied to the target qubit.

This gate is represented by a block-diagonal matrix, where each block is a
2x2 unitary:

    [[U_0, 0,   ....,        0],
     [0,   U_1, ....,        0],
                .
                    .
     [0,   0,  ...., U_(2^k-1)]]
"""

# pylint: disable=unused-import
from qiskit.circuit.library.generalized_gates.uc import UCGate
from qiskit.utils.deprecation import _deprecate_extension

_deprecate_extension("UCGate")
