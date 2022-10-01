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

"""Barrier instruction."""

from qiskit.exceptions import QiskitError
from .instruction import Instruction


class Barrier(Instruction):
    """Barrier instruction.
        The barrier acts as a directive for circuit compilation to separate pieces of a circuit
        so that any optimizations or re-writes are constrained to only act between barriers
        (and if there are no barriers they act on the whole circuit). This only comes into play
        when using the transpile or execute functions in Qiskit (execute includes a transpile step)"""

    _directive = True

    def __init__(self, num_qubits):
        """Create new barrier instruction."""
        super().__init__("barrier", num_qubits, 0, [])

    def inverse(self):
        """Special case. Return self."""
        return Barrier(self.num_qubits)

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise QiskitError("Barriers are compiler directives and cannot be conditional.")
