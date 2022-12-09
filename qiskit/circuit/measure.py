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
Quantum measurement in the computational basis.
"""

from __future__ import annotations

import warnings

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.exceptions import CircuitError


# Measure class kept for backwards compatibility, and repurposed
# as a common parent class for all measurement instructions.
class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    basis: str | None = None

    def __init__(self):
        """Create new measurement instruction."""
        super().__init__("measure", 1, 1, [])

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError("register size error")


class MeasureX(Measure):
    """Quantum measurement in the X basis."""

    basis: str | None = "X"

    def __init__(self):
        """Create new X measurement instruction."""
        # pylint: disable=bad-super-call
        super(Measure, self).__init__("measure_x", 1, 1, [])

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure_z(0, 0)
        qc.h(0)

        self.definition = qc


class MeasureY(Measure):
    """Quantum measurement in the Y basis."""

    basis: str | None = "Y"

    def __init__(self):
        """Create new Y measurement instruction."""
        # pylint: disable=bad-super-call
        super(Measure, self).__init__("measure_y", 1, 1, [])

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.sdg(0)
        qc.measure_x(0, 0)
        qc.s(0)

        self.definition = qc


class MeasureZ(Measure):
    """Quantum Z measurement in the Z basis."""

    basis: str | None = "Z"

    def __init__(self):
        """Create new measurement instruction."""
        # pylint: disable=bad-super-call
        super(Measure, self).__init__("measure_z", 1, 1, [])

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)

        self.definition = qc


def measure(circuit, qubit, clbit):
    """Measure a quantum bit into classical bit.

    .. deprecated:: Qiskit Terra 0.19
        Use :meth:`.QuantumCircuit.measure` instead, either by calling
        ``circuit.measure(qubit, clbit)``, or if a full function is required, then
        ``QuantumCircuit.measure(circuit, qubit, clbit)``.

    Args:
        circuit (QuantumCircuit): the quantum circuit to attach the measurement to.
        qubit (Union[Qubit, int]): the quantum bit to measure
        clbit (Union[Clbit, int]): the classical bit to store the measurement result in.

    Returns:
        .InstructionSet: a handle to the created instruction.

    Raises:
        CircuitError: if either bit is not in the circuit, or is in a bad format.
    """
    warnings.warn(
        "The loose 'measure' function is deprecated as of Qiskit Terra 0.19, and will be removed"
        " in a future release.  Instead, you should call 'circuit.measure(qubit, clbit)', or if you"
        " need a function, you can do `QuantumCircuit.measure(circuit, qubit, clbit)'.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return circuit.measure(qubit, clbit)
