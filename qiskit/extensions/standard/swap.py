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
from qiskit.extensions.standard.cx import CXGate
from qiskit.extensions.standard.cx import CCXGate


class SwapGate(Gate):
    """Swap gate."""

    def __init__(self):
        """Create new SWAP gate."""
        super().__init__('swap', 2, [])

    def _define(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return FredkinGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate."""
        return SwapGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Swap gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=complex)


def swap(self, qubit1, qubit2):
    """Apply Swap from qubit1 to qubit2."""
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
    """The controlled-swap gate, also called Fredkin gate."""

    def __init__(self):
        """Create new CSwap gate."""
        super().__init__('cswap', 3, [], num_ctrl_qubits=1)
        self.base_gate = SwapGate
        self.base_gate_name = 'swap'

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        definition = []
        q = QuantumRegister(3, 'q')
        rule = [
            (CXGate(), [q[2], q[1]], []),
            (CCXGate(), [q[0], q[1], q[2]], []),
            (CXGate(), [q[2], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CSwapGate()  # self-inverse


class FredkinGate(CSwapGate, metaclass=CSwapMeta):
    """The deprecated CSwapGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class FredkinGate is deprecated as of 0.12.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CSwapGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


def cswap(self, ctl, tgt1, tgt2):
    """Apply CSwap to circuit."""
    return self.append(CSwapGate(), [ctl, tgt1, tgt2], [])


# support both cswap and fredkin as methods of QuantumCircuit
QuantumCircuit.cswap = cswap
QuantumCircuit.fredkin = cswap
