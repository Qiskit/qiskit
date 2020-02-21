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

"""
T=sqrt(S) phase gate or its inverse.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class TGate(Gate):
    """T Gate: pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create new T gate."""
        super().__init__('t', 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TdgGate()

    def to_matrix(self):
        """Return a numpy.array for the T gate."""
        return numpy.array([[1, 0],
                            [0, (1 + 1j) / numpy.sqrt(2)]], dtype=complex)


class TdgGate(Gate):
    """Tdg Gate: -pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create a new Tdg gate."""
        super().__init__('tdg', 1, [], label=label)

    def _define(self):
        """
        gate tdg a { u1(pi/4) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(-pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TGate()

    def to_matrix(self):
        """Return a numpy.array for the inverse T gate."""
        return numpy.array([[1, 0],
                            [0, (1 - 1j) / numpy.sqrt(2)]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def t(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply T gate to a specified qubit (qubit).
    A T gate implements a pi/4 rotation of a qubit state vector about the
    z axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.t(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.t import TGate
            TGate().to_matrix()
    """
    return self.append(TGate(), [qubit], [])


@deprecate_arguments({'q': 'qubit'})
def tdg(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply Tdg gate to a specified qubit (qubit).
    A Tdg gate implements a -pi/4 rotation of a qubit state vector about the
    z axis of the Bloch sphere. It is the inverse of T-gate.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.tdg(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.t import TdgGate
            TdgGate().to_matrix()
    """
    return self.append(TdgGate(), [qubit], [])


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
