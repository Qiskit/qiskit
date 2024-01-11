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

"""
Qubit reset to computational zero.
"""

import warnings

from qiskit.circuit.instruction import Instruction


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self):
        """Create new reset instruction."""
        super().__init__("reset", 1, 0, [])

    def broadcast_arguments(self, qargs, cargs):
        for qarg in qargs[0]:
            yield [qarg], []


def reset(circuit, qubit):
    """Reset a quantum bit on a circuit.

    .. deprecated:: Qiskit Terra 0.19
        Use :meth:`.QuantumCircuit.reset` instead, either by calling ``circuit.reset(qubit)``, or if
        a full function is required, then ``QuantumCircuit.reset(circuit, qubit)``.

    Args:
        circuit (QuantumCircuit): the quantum circuit to attach the reset operation to.
        qubit (Union[Qubit, int]): the quantum bit to reset

    Returns:
        .InstructionSet: a handle to the created instruction.

    Raises:
        CircuitError: if the qubit is not in the circuit, or is in a bad format.
    """
    warnings.warn(
        "The loose 'reset' function is deprecated as of Qiskit Terra 0.19, and will be removed"
        " in a future release.  Instead, you should call 'circuit.reset(qubit)', or if you"
        " need a function, you can do `QuantumCircuit.reset(circuit, qubit)'.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return circuit.reset(qubit)
