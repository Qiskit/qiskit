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
Assertion of superposition states.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.circuit.assertmanager import AssertManager
from qiskit.circuit.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
#from numpy import math
from random import randint
from scipy.stats import chisquare

class AssertSuperposition(Asserts):
    """Assertion of superposition states
       and Quantum measurement in the computational basis."""
    def __init__(self, pcrit):
        super().__init__()
        self._type = "Superposition"
        self._pcrit = pcrit

    def stat_test(self, counts):
        vals_list = list(counts.values())
        numzeros = len(counts) - 2**len(list(counts)[0])
        vals_list.extend([0]*numzeros)
        print("vals_list = ")
        print(vals_list)
        chisq, pval = chisquare(vals_list)
        print("chisq, pval = ")
        print(chisq, pval)
        if pval <= self._pcrit or chisq == 0: #math.isnan(pval):
            passed = False
        else:
            passed = True
        print("Superposition: chisq, pval, passed = ")
        print(chisq, pval, passed)
        return (chisq, pval, passed)


def assertsuperposition(self, pcrit, qubit, cbit):
    """Create superposition assertion

    Args:
        pcrit: float
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
    AssertManager.StatOutputs[theClone.name] = {"type":"Superposition"}
    theClone.append(AssertSuperposition(pcrit), [qubit], [cbit])
    return theClone

QuantumCircuit.assertsuperposition = assertsuperposition
