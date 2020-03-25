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
Quantum measurement in the computational basis.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from numpy import pi


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self):
        """Create new measurement instruction."""
        super().__init__("measure", 1, 1, [])

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError('register size error')


def measure(self, qubit, cbit):
    """Measure quantum bit into classical bit (tuples).

    Args:
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.Instruction: the attached measure instruction.

    Raises:
        CircuitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    return self.append(Measure(), [qubit], [cbit])

QuantumCircuit.measure = measure


def measure_x(self, qubit, cbit):
    m_x = QuantumCircuit(1,1,name='Mx')
    m_x.h(0)
    m_x.measure(0,0)
    return self.append(m_x.to_instruction(), [qubit], [cbit])

def measure_y(self, qubit, cbit):
    m_y = QuantumCircuit(1,1,name='My')
    m_y.rx(pi/2,0)
    m_y.measure(0,0)
    return self.append(m_y.to_instruction(), [qubit], [cbit])

QuantumCircuit.measure_x = measure_x
QuantumCircuit.measure_y = measure_y
QuantumCircuit.measure_z = measure
