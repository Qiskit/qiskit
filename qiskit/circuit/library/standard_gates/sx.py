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

"""Sqrt(X) and C-Sqrt(X) gates."""

from typing import Optional, Union
import numpy
from qiskit.qasm import pi
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class SXGate(Gate):
    r"""The single-qubit Sqrt(X) gate (:math:`\sqrt{X}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sx` method.

    **Matrix Representation:**

    .. math::

        \sqrt{X} = \frac{1}{2} \begin{pmatrix}
                1 + i & 1 - i \\
                1 - i & 1 + i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌────┐
        q_0: ┤ √X ├
             └────┘

    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(\pi/2)` and :math:`\sqrt{X}`.

        .. math::

            RX(\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & -i \\
                        -i & 1
                      \end{pmatrix}
                    = e^{-i pi/4} \sqrt{X}

    """

    def __init__(self, label: Optional[str] = None):
        """Create new SX gate."""
        super().__init__("sx", 1, [], label=label)

    def _define(self):
        """
        gate sx a { rz(-pi/2) a; h a; rz(-pi/2); }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .s import SdgGate
        from .h import HGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=pi / 4)
        rules = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], []), (SdgGate(), [q[0]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def inverse(self):
        """Return inverse SX gate (i.e. SXdg)."""
        return SXdgGate()

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a (multi-)controlled-SX gate.

        One control returns a CSX gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CSXGate(label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the SX gate."""
        return numpy.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=dtype) / 2


class SXdgGate(Gate):
    r"""The inverse single-qubit Sqrt(X) gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sxdg` method.

    .. math::

        \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix}
                1 - i & 1 + i \\
                1 + i & 1 - i
            \end{pmatrix}


    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(-\pi/2)` and :math:`\sqrt{X}^{\dagger}`.

        .. math::

            RX(-\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & i \\
                        i & 1
                      \end{pmatrix}
                    = e^{-i pi/4} \sqrt{X}^{\dagger}

    """

    def __init__(self, label: Optional[str] = None):
        """Create new SXdg gate."""
        super().__init__("sxdg", 1, [], label=label)

    def _define(self):
        """
        gate sxdg a { rz(pi/2) a; h a; rz(pi/2); }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .s import SGate
        from .h import HGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=-pi / 4)
        rules = [(SGate(), [q[0]], []), (HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def inverse(self):
        """Return inverse SXdg gate (i.e. SX)."""
        return SXGate()

    def __array__(self, dtype=None):
        """Return a numpy.array for the SXdg gate."""
        return numpy.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=dtype) / 2


class CSXGate(ControlledGate):
    r"""Controlled-√X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csx` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴──┐
        q_1: ┤ √X ├
             └────┘

    **Matrix representation:**

    .. math::

        C\sqrt{X} \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + \sqrt{X} \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & (1 + i) / 2 & 0 & (1 - i) / 2 \\
                0 & 0 & 1 & 0 \\
                0 & (1 - i) / 2 & 0 & (1 + i) / 2
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be `q_1`. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌────┐
            q_0: ┤ √X ├
                 └─┬──┘
            q_1: ──■──

        .. math::

            C\sqrt{X}\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes \sqrt{X} =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & (1 + i) / 2 & (1 - i) / 2 \\
                    0 & 0 & (1 - i) / 2 & (1 + i) / 2
                \end{pmatrix}

    """
    # Define class constants. This saves future allocation time.
    _matrix1 = numpy.array(
        [
            [1, 0, 0, 0],
            [0, (1 + 1j) / 2, 0, (1 - 1j) / 2],
            [0, 0, 1, 0],
            [0, (1 - 1j) / 2, 0, (1 + 1j) / 2],
        ]
    )
    _matrix0 = numpy.array(
        [
            [(1 + 1j) / 2, 0, (1 - 1j) / 2, 0],
            [0, 1, 0, 0],
            [(1 - 1j) / 2, 0, (1 + 1j) / 2, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create new CSX gate."""
        super().__init__(
            "csx", 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=SXGate()
        )

    def _define(self):
        """
        gate csx a,b { h b; cu1(pi/2) a,b; h b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .u1 import CU1Gate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[1]], []), (CU1Gate(pi / 2), [q[0], q[1]], []), (HGate(), [q[1]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the CSX gate."""
        mat = self._matrix1 if self.ctrl_state else self._matrix0
        if dtype:
            return numpy.asarray(mat, dtype=dtype)
        return mat
