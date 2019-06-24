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
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from random import randint


class AssertProduct(Measure):
    """Quantum measurement in the computational basis."""
    ExpectedProductStates = {}
    def __init__(self):
        """Create new measurement instruction."""
        super().__init__()


def assertproduct(self, qubit0, cbit0, qubit1, cbit1):
    """Create product assertion

    Args:
        qubit0 (QuantumRegister|list|tuple): quantum register
        cbit0 (ClassicalRegister|list|tuple): classical register
        qubit1 (QuantumRegister|list|tuple): quantum register
        cbit1 (ClassicalRegister|list|tuple): classical register


    Returns:
        qiskit.QuantumCircuit: copy of quantum circuit at the assert point.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    randString = str(randint(0, 1000000000))
    theClone = self.copy("breakpoint"+randString)
    AssertProduct.ExpectedProductStates[theClone.name] = [qubit0, cbit0, qubit1, cbit1]
    theClone.append(AssertProduct(), [qubit0.extend(qubit1)], [cbit0.extend(cbit1)])
    return theClone

QuantumCircuit.assertproduct = assertproduct

