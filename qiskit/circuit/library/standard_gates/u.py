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

"""Two-pulse single-qubit gate."""

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.exceptions import CircuitError


class UGate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.

    Implemented using two X90 pulses on IBM Quantum systems:

    .. math::
        U(\theta, \phi, \lambda) =
            RZ(\phi - \pi/2) RX(\pi/2) RZ(\pi - \theta) RX(\pi/2) RZ(\lambda - \pi/2)

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤ U(ϴ,φ,λ) ├
             └──────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos(\th)          & -e^{i\lambda}\sin(\th) \\
                e^{i\phi}\sin(\th) & e^{i(\phi+\lambda)}\cos(\th)
            \end{pmatrix}

    **Examples:**

    .. math::

        U\left(\theta, -\frac{\pi}{2}, \frac{\pi}{2}\right) = RX(\theta)

    .. math::

        U(\theta, 0, 0) = RY(\theta)
    """

    def __init__(self, theta, phi, lam, label=None):
        """Create new U gate."""
        super().__init__("u", 1, [theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return UGate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (multi-)controlled-U gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CUGate(
                self.params[0],
                self.params[1],
                self.params[2],
                0,
                label=label,
                ctrl_state=ctrl_state,
            )
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the U gate."""
        theta, phi, lam = (float(param) for param in self.params)
        return numpy.array(
            [
                [numpy.cos(theta / 2), -numpy.exp(1j * lam) * numpy.sin(theta / 2)],
                [
                    numpy.exp(1j * phi) * numpy.sin(theta / 2),
                    numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2),
                ],
            ],
            dtype=dtype,
        )


class CUGate(ControlledGate):
    r"""Controlled-U gate (4-parameter two-qubit gate).

    This is a controlled version of the U gate (generic single qubit rotation),
    including a possible global phase :math:`e^{i\gamma}` of the U gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──────■──────
             ┌─────┴──────┐
        q_1: ┤ U(ϴ,φ,λ,γ) ├
             └────────────┘

    **Matrix representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        CU(\theta, \phi, \lambda, \gamma)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| +
            e^{i\gamma} U(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0                           & 0 & 0 \\
                0 & e^{i\gamma}\cos(\th)        & 0 & -e^{i(\gamma + \lambda)}\sin(\th) \\
                0 & 0                           & 1 & 0 \\
                0 & e^{i(\gamma+\phi)}\sin(\th) & 0 & e^{i(\gamma+\phi+\lambda)}\cos(\th)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌────────────┐
            q_0: ┤ U(ϴ,φ,λ,γ) ├
                 └─────┬──────┘
            q_1: ──────■───────

        .. math::

            CU(\theta, \phi, \lambda, \gamma)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I +
                e^{i\gamma}|1\rangle\langle 1| \otimes U(\theta,\phi,\lambda) =
                \begin{pmatrix}
                    1 & 0 & 0                             & 0 \\
                    0 & 1 & 0                             & 0 \\
                    0 & 0 & e^{i\gamma} \cos(\th)         & -e^{i(\gamma + \lambda)}\sin(\th) \\
                    0 & 0 & e^{i(\gamma + \phi)}\sin(\th) & e^{i(\gamma + \phi+\lambda)}\cos(\th)
                \end{pmatrix}
    """

    def __init__(self, theta, phi, lam, gamma, label=None, ctrl_state=None):
        """Create new CU gate."""
        super().__init__(
            "cu",
            2,
            [theta, phi, lam, gamma],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=UGate(theta, phi, lam),
        )

    def _define(self):
        """
        gate cu(theta,phi,lambda,gamma) c, t
        { phase(gamma) c;
          phase((lambda+phi)/2) c;
          phase((lambda-phi)/2) t;
          cx c,t;
          u(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u(theta/2,phi,0) t;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[3], 0)
        qc.p((self.params[2] + self.params[1]) / 2, 0)
        qc.p((self.params[2] - self.params[1]) / 2, 1)
        qc.cx(0, 1)
        qc.u(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2, 1)
        qc.cx(0, 1)
        qc.u(self.params[0] / 2, self.params[1], 0, 1)
        self.definition = qc

    def inverse(self):
        r"""Return inverted CU gate.

        :math:`CU(\theta,\phi,\lambda,\gamma)^{\dagger} = CU(-\theta,-\phi,-\lambda,-\gamma)`)
        """
        return CUGate(
            -self.params[0],
            -self.params[2],
            -self.params[1],
            -self.params[3],
            ctrl_state=self.ctrl_state,
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the CU gate."""
        theta, phi, lam, gamma = (float(param) for param in self.params)
        cos = numpy.cos(theta / 2)
        sin = numpy.sin(theta / 2)
        a = numpy.exp(1j * gamma) * cos
        b = -numpy.exp(1j * (gamma + lam)) * sin
        c = numpy.exp(1j * (gamma + phi)) * sin
        d = numpy.exp(1j * (gamma + phi + lam)) * cos
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, a, 0, b], [0, 0, 1, 0], [0, c, 0, d]], dtype=dtype
            )
        else:
            return numpy.array(
                [[a, 0, b, 0], [0, 1, 0, 0], [c, 0, d, 0], [0, 0, 0, 1]], dtype=dtype
            )

    @property
    def params(self):
        """Get parameters from base_gate.

        Returns:
            list: List of gate parameters.

        Raises:
            CircuitError: Controlled gate does not define a base gate
        """
        if self.base_gate:
            # CU has one additional parameter to the U base gate
            return self.base_gate.params + self._params
        else:
            raise CircuitError("Controlled gate does not define base gate " "for extracting params")

    @params.setter
    def params(self, parameters):
        """Set base gate parameters.

        Args:
            parameters (list): The list of parameters to set.

        Raises:
            CircuitError: If controlled gate does not define a base gate.
        """
        # CU has one additional parameter to the U base gate
        self._params = [parameters[-1]]
        if self.base_gate:
            self.base_gate.params = parameters[:-1]
        else:
            raise CircuitError("Controlled gate does not define base gate " "for extracting params")
