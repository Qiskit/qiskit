# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Example showing how to draw a quantum circuit using Qiskit.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit import CircuitInstruction, Qubit, QuantumRegister
circuit = QuantumCircuit()


def instructions_bell_circuit():
    instructions_list = []
    instruction = CircuitInstruction(operation=Instruction(name='h', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=())
    instructions_list.append(instruction)
    instruction = CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())
    instructions_list.append(instruction)
    return instructions_list
    

print(circuit.from_instructions(instructions_bell_circuit()))

def build_bell_circuit():
    qcirc = QuantumCircuit(2,2)
    qcirc.h(0)
    qcirc.cx(0,1)
    return qcirc

# Read instructions from the circuit
print(circuit.from_instructions(build_bell_circuit()))
