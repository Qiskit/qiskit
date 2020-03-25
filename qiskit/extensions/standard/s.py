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
S and Sdg gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class SGate(Gate):
    r"""Single qubit S gate (Z**0.5).

    It induces a :math:`\pi/2` phase, and is sometimes called the P gate (phase).

    This is a Clifford gate and a square-root of Pauli-Z.

    **Matrix Representation:**

    .. math::

        S = \begin{pmatrix}
                1 & 0 \\
                0 & i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ S ├
             └───┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """
    def __init__(self, label=None):
        """Create a new S gate."""
        super().__init__('s', 1, [], label=label)

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse of S (SdgGate)."""
        return SdgGate()

    def to_matrix(self):
        """Return a numpy.array for the S gate."""
        return numpy.array([[1, 0],
                            [0, 1j]], dtype=complex)


class SdgGate(Gate):
    r"""Single qubit S-adjoint gate (~Z**0.5).

    It induces a :math:`-\pi/2` phase.

    This is a Clifford gate and a square-root of Pauli-Z.

    **Matrix Representation:**

    .. math::

        Sdg = \begin{pmatrix}
                1 & 0 \\
                0 & -i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Sdg ├
             └─────┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """

    def __init__(self, label=None):
        """Create a new Sdg gate."""
        super().__init__('sdg', 1, [], label=label)

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(-pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse of Sdg (SGate)."""
        return SGate()

    def to_matrix(self):
        """Return a numpy.array for the Sdg gate."""
        return numpy.array([[1, 0],
                            [0, -1j]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def s(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply :class:`~qiskit.extensions.standard.SGate`.
    """
    return self.append(SGate(), [qubit], [])


@deprecate_arguments({'q': 'qubit'})
def sdg(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.SdgGate`.
    """
    return self.append(SdgGate(), [qubit], [])


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
