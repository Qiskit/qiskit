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
from qiskit.circuit.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from random import randint
from scipy.stats import chisquare

class AssertSuperposition(Asserts):
    """Quantum measurement in the computational basis."""
    def __init__(self):
        """Create new measurement instruction."""
        super().__init__()


def assertsuperposition(self, qubit, cbit):
    """Create superposition assertion.

    Args:
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.QuantumCircuit: copy of quantum circuit at the assert point.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    randString = str(randint(0, 1000000000))
    theClone = self.copy("breakpoint"+randString)
    theClone.append(AssertSuperposition(), [qubit], [cbit])    
    Asserts.StatOutputs[theClone.name] = {"type": "Superposition"}
    return theClone

def stat_test(self, counts):
    chisq, p = chisquare(counts.values)
    Asserts.StatOutput[self.name] = {"type" : "Superposition", "chisq": chisq, "p": p}
    return 0

QuantumCircuit.assertsuperposition = assertsuperposition

