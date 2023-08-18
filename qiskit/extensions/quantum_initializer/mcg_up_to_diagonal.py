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

# pylint: disable=unused-variable

"""
Multi controlled single-qubit unitary up to diagonal.
"""

# ToDo: This code should be merged wth the implementation of MCGs
# ToDo: (introducing a decomposition mode "up_to_diagonal").

from qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal import MCGupDiag as NewMCGupDiag
from qiskit.utils.deprecation import deprecate_func

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class MCGupDiag(NewMCGupDiag):
    """
    Decomposes a multi-controlled gate u up to a diagonal d acting on the control and target qubit
    (but not on the  ancilla qubits), i.e., it implements a circuit corresponding to a unitary u'
    such that u=d.u'.
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.MCGupDiagonal."
    )
    def __init__(self, gate, num_controls, num_ancillas_zero, num_ancillas_dirty):
        """Initialize a multi controlled gate.

        Args:
            gate (ndarray): 2*2 unitary (given as a (complex) ndarray)
            num_controls (int): number of control qubits
            num_ancillas_zero (int): number of ancilla qubits that start in the state zero
            num_ancillas_dirty (int): number of ancilla qubits that are allowed to start in an
                arbitrary state
        Raises:
            QiskitError: if the input format is wrong; if the array gate is not unitary
        """
        super().__init__(gate, num_controls, num_ancillas_zero, num_ancillas_dirty)
