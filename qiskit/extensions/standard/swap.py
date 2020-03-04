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
Swap gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


class SwapGate(Gate):
    r"""Swap gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{Swap}}
            = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new Swap gate."""
        super().__init__("swap", 2, [],
                         phase=phase, label=label)

    def _define(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (CXGate(phase=self.phase), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (CXGate(), [q[0], q[1]], [])
        ]

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            if num_ctrl_qubits == 1:
                return CSwapGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return SwapGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the SWAP gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=complex)


def swap(self, qubit1, qubit2):
    """Apply SWAP gate to a pair specified qubits (qubit1, qubit2).
    The SWAP gate canonically swaps the states of two qubits.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.swap(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.swap import SwapGate
            SwapGate().to_matrix()
    """
    return self.append(SwapGate(), [qubit1, qubit2], [])


QuantumCircuit.swap = swap


class CSwapMeta(type):
    """A Metaclass to ensure that CSwapGate and FredkinGate are of the same type.

    Can be removed when FredkinGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CSwapGate, FredkinGate}  # pylint: disable=unidiomatic-typecheck


class CSwapGate(ControlledGate, metaclass=CSwapMeta):
    r"""The controlled-swap gate, also called Fredkin gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{CSwap}} =&
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{Swap}} \otimes |1 \rangle\!\langle 1| \\
            =&
            \begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{bmatrix}
    """
    def __init__(self, phase=0, label=None):
        """Create new CSwap gate."""
        super().__init__("cswap", 3, [], phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = SwapGate()

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        from qiskit.extensions.standard.x import CXGate
        from qiskit.extensions.standard.x import CCXGate
        q = QuantumRegister(3, 'q')
        self.definition = [
            (CXGate(phase=self.phase), [q[2], q[1]], []),
            (CCXGate(), [q[0], q[1], q[2]], []),
            (CXGate(), [q[2], q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CSwapGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the Fredkin (CSWAP) gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)



class FredkinGate(CSwapGate, metaclass=CSwapMeta):
    """The deprecated CSwapGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class FredkinGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CSwapGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt1': 'target_qubit1',
                      'tgt2': 'target_qubit2'})
def cswap(self, control_qubit, target_qubit1, target_qubit2,
          *, ctl=None, tgt1=None, tgt2=None):  # pylint: disable=unused-argument
    """Apply Fredkin (CSWAP) gate from a specified control (control_qubit) to target1
    (target_qubit1) and target2 (target_qubit2) qubits. The CSWAP gate is canonically
    used to swap the qubit states of target1 and target2 when the control qubit is in
    state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(3)
            circuit.cswap(0,1,2)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.swap import CSwapGate
            CSwapGate().to_matrix()
    """
    return self.append(CSwapGate(), [control_qubit, target_qubit1, target_qubit2], [])


# support both cswap and fredkin as methods of QuantumCircuit
QuantumCircuit.cswap = cswap
QuantumCircuit.fredkin = cswap
