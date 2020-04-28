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

"""T and Tdg gate."""

import numpy
from qiskit.qasm import pi
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class TGate(Gate):
    r"""Single qubit T gate (Z**0.25).

    It induces a :math:`\pi/4` phase, and is sometimes called the pi/8 gate
    (because of how the RZ(\pi/4) matrix looks like).

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    **Matrix Representation:**

    .. math::

        T = \begin{pmatrix}
                1 & 0 \\
                0 & 1+i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ T ├
             └───┘

    Equivalent to a :math:`\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label=None):
        """Create new T gate."""
        super().__init__('t', 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        from .u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse T gate (i.e. Tdg)."""
        return TdgGate()

    def to_matrix(self):
        """Return a numpy.array for the T gate."""
        return numpy.array([[1, 0],
                            [0, (1 + 1j) / numpy.sqrt(2)]], dtype=complex)


class TdgGate(Gate):
    r"""Single qubit T-adjoint gate (~Z**0.25).

    It induces a :math:`-\pi/4` phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    **Matrix Representation:**

    .. math::

        Tdg = \begin{pmatrix}
                1 & 0 \\
                0 & 1-i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Tdg ├
             └─────┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """

    def __init__(self, label=None):
        """Create new Tdg gate."""
        super().__init__('tdg', 1, [], label=label)

    def _define(self):
        """
        gate tdg a { u1(pi/4) a; }
        """
        from .u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(-pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse Tdg gate (i.e. T)."""
        return TGate()

    def to_matrix(self):
        """Return a numpy.array for the inverse T gate."""
        return numpy.array([[1, 0],
                            [0, (1 - 1j) / numpy.sqrt(2)]], dtype=complex)
