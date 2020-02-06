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
S=diag(1,i) Clifford phase gate or its inverse.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class SGate(Gate):
    """S=diag(1,i) Clifford phase gate."""

    def __init__(self, label=None):
        """Create new S gate."""
        super().__init__("s", 1, [], label=label)

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SdgGate()

    def to_matrix(self):
        """Return a Numpy.array for the S gate."""
        return numpy.array([[1, 0],
                            [0, 1j]], dtype=complex)


class SdgGate(Gate):
    """Sdg=diag(1,-i) Clifford adjoint phase gate."""

    def __init__(self, label=None):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], label=label)

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(-pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SGate()

    def to_matrix(self):
        """Return a Numpy.array for the Sdg gate."""
        return numpy.array([[1, 0],
                            [0, -1j]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def s(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply S gate to a specified qubit (qubit).
    An S gate implements a pi/2 rotation of the qubit state vector about the
    z axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.s(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.s import SGate
            SGate().to_matrix()
    """
    return self.append(SGate(), [qubit], [])


@deprecate_arguments({'q': 'qubit'})
def sdg(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply Sdg gate to a specified qubit (qubit).
    An Sdg gate implements a -pi/2 rotation of the qubit state vector about the
    z axis of the Bloch sphere. It is the inverse of S gate.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.sdg(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.s import SdgGate
            SdgGate().to_matrix()
    """
    return self.append(SdgGate(), [qubit], [])


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
