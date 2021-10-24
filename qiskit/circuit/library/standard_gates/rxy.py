# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XY-rotation gate."""

from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


class RXYGate(Gate):
    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new RXY gate."""
        super().__init__("rxy", 2, [theta], label=label)

    def _define(self):
        """
        gate rxy(theta) a, b { h b; cx b, a; ry(theta) a; cx b, a; h b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        from .ry import RYGate

        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (RYGate(theta), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (HGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse RXY gate (i.e. with the negative rotation angle)."""
        return RXYGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RXY gate."""
        import numpy

        half_theta = float(self.params[0]) / 2
        cos = numpy.cos(half_theta)
        sin = numpy.sin(half_theta)
        return numpy.array(
            [[cos, 0, 0, -sin], [0, cos, sin, 0], [0, -sin, cos, 0], [sin, 0, 0, cos]],
            dtype=dtype,
        )
