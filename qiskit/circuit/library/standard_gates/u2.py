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

"""One-pulse single-qubit gate."""

import numpy
from qiskit.qasm import pi
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class U2Gate(Gate):
    r"""Single-qubit rotation about the X+Z axis.

    Implemented using one X90 pulse on IBM Quantum systems:

    .. math::
        U2(\phi, \lambda) = RZ(\phi).RY(\frac{\pi}{2}).RZ(\lambda)

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤ U2(φ,λ) ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        U2(\phi, \lambda) = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1          & -e^{i\lambda} \\
                e^{i\phi} & e^{i(\phi+\lambda)}
            \end{pmatrix}

    **Examples:**

    .. math::

        U2(0, \pi) = H
        U2(0, 0) = RY(\pi/2)
        U2(-\pi/2, \pi/2) = RX(\pi/2)
    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.
    """

    def __init__(self, phi, lam, label=None):
        """Create new U2 gate."""
        super().__init__('u2', 1, [phi, lam], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi / 2, self.params[0], self.params[1]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted U2 gate.

        :math:`U2(\phi, \lambda)^{\dagger} =U2(-\lambda-\pi, -\phi+\pi)`)
        """
        return U2Gate(-self.params[1] - pi, -self.params[0] + pi)

    def to_matrix(self):
        """Return a Numpy.array for the U2 gate."""
        isqrt2 = 1 / numpy.sqrt(2)
        phi, lam = self.params
        phi, lam = float(phi), float(lam)
        return numpy.array([
            [
                isqrt2,
                -numpy.exp(1j * lam) * isqrt2
            ],
            [
                numpy.exp(1j * phi) * isqrt2,
                numpy.exp(1j * (phi + lam)) * isqrt2
            ]
        ], dtype=complex)
