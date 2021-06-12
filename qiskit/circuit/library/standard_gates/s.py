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

"""The S and Sdg gate."""

import numpy
from qiskit.qasm import pi
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


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
        """Create new S gate."""
        super().__init__("s", 1, [], label=label)

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse of S (SdgGate)."""
        return SdgGate()

    def __array__(self, dtype=None):
        """Return a numpy.array for the S gate."""
        return numpy.array([[1, 0], [0, 1j]], dtype=dtype)


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
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], label=label)

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse of Sdg (SGate)."""
        return SGate()

    def __array__(self, dtype=None):
        """Return a numpy.array for the Sdg gate."""
        return numpy.array([[1, 0], [0, -1j]], dtype=dtype)
