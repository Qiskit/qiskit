# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum measurement in the Pauli basis.
"""
from typing import List
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class PauliMeasure(Instruction):
    """Quantum measurement in the Pauli basis."""

    def __init__(self, params: List):
        """Create new Pauli measurement instruction."""
        super().__init__("measure", 1, 1, params)

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

    def _define(self):
        """Decompose to 1-qubit measurements in the computational basis"""
        # pylint: disable=cyclic-import
        from qiskit import QuantumCircuit

        pauli_string = self.params[0]
        nqubits = len(pauli_string)
        qc = QuantumCircuit(nqubits, nqubits,
                            name='pauli_measure_' + pauli_string)
        
        for i, p enumerate(reversed(pauli_string)):
            if p == 'I':
                continue
            if p == 'X':
                qc.h(i)
            if p == 'Y':
                qc.s(i)
                qc.z(i)
                qc.h(i)
            qc.measure(i, i)
                
        self.definition = qc

     def validate_parameter(self, parameter):
        if isinstance(parameter, str):
            if all([c in ["I", "X", "Y", "Z"] for c in parameter]):
                return parameter
            else:
                raise CircuitError("Parameter string {0} should contain only "
                                   "'I', 'X', 'Y', 'Z' characters")
        else:
            raise CircuitError("Parameter {0} should be a string of "
                               "'I', 'X', 'Y', 'Z' characters")
        

def pauli_measure(self, qubit, cbit, params):
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
    return self.append(PauliMeasure(), [qubit], [cbit])


QuantumCircuit.pauli_measure = pauli_measure
