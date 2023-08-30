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

"""
Implementation of the abstract class UCPauliRotGate for uniformly controlled
(also called multiplexed) single-qubit rotations around the Z-axes
(i.e., uniformly controlled R_z rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_z(a_i) is applied to the target qubit.
"""

# pylint: disable=unused-import
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate
from qiskit.utils.deprecation import _deprecate_extension

_deprecate_extension("UCRZGate")
