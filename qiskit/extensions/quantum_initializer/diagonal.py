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

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Decomposes a diagonal matrix into elementary gates using the method described in Theorem 7 in
"Synthesis of Quantum Logic Circuits" by Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf).
"""
import math

from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.generalized_gates.diagonal import DiagonalGate as NewDiagonalGate
from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_func


class DiagonalGate(NewDiagonalGate):
    """
    diag = list of the 2^k diagonal entries (for a diagonal gate on k qubits). Must contain at
    least two entries.
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.DiagonalGate."
    )
    def __init__(self, diag):
        """
        Args:
            diag (list): list of the 2^k diagonal entries (for a diagonal gate on k qubits).
                Must contain at least two entries
        """
        super().__init__(diag)


@deprecate_func(
    since="0.45.0",
    additional_msg="Instead, compose the circuit with a qiskit.circuit.library.Diagonal circuit.",
)
def diagonal(self, diag, qubit):
    """Attach a diagonal gate to a circuit.

    The decomposition is based on Theorem 7 given in "Synthesis of Quantum Logic Circuits" by
    Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf).

    Args:
        diag (list): list of the 2^k diagonal entries (for a diagonal gate on k qubits).
            Must contain at least two entries
        qubit (QuantumRegister|list): list of k qubits the diagonal is
            acting on (the order of the qubits specifies the computational basis in which the
            diagonal gate is provided: the first element in diag acts on the state where all
            the qubits in q are in the state 0, the second entry acts on the state where all
            the qubits q[1],...,q[k-1] are in the state zero and q[0] is in the state 1,
            and so on)

    Returns:
        QuantumCircuit: the diagonal gate which was attached to the circuit.

    Raises:
        QiskitError: if the list of the diagonal entries or the qubit list is in bad format;
            if the number of diagonal entries is not 2^k, where k denotes the number of qubits
    """

    if isinstance(qubit, QuantumRegister):
        qubit = qubit[:]
    # Check if q has type "list"
    if not isinstance(qubit, list):
        raise QiskitError(
            "The qubits must be provided as a list (also if there is only one qubit)."
        )
    # Check if diag has type "list"
    if not isinstance(diag, list):
        raise QiskitError("The diagonal entries are not provided in a list.")
    num_action_qubits = math.log2(len(diag))
    if not len(qubit) == num_action_qubits:
        raise QiskitError(
            "The number of diagonal entries does not correspond to the number of qubits."
        )

    gate = NewDiagonalGate(diag)

    return self.append(gate, qubit)


QuantumCircuit.diagonal = diagonal
