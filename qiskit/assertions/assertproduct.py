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
Assertion of product states.
"""
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.assertions.assertmanager import AssertManager
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chi2_contingency

class AssertProduct(Asserts):
    """
        Assertion of product states and quantum measurement
        in the computational basis.
    """
    def __init__(self, pcrit, qubit0, cbit0, qubit1, cbit1): 
        super().__init__()
        self._type = "Product"
        self._pcrit = pcrit
        self._qubit0 = AssertManager.syntax4measure(qubit0)
        self._cbit0 = AssertManager.syntax4measure(cbit0)
        self._qubit1 = AssertManager.syntax4measure(qubit1)
        self._cbit1 = AssertManager.syntax4measure(cbit1)
        self._qubit = self._qubit0 + self._qubit1
        self._cbit = self._cbit0 + self._cbit1

    def stat_test(self, counts):
        # chi-squared contingency table test to assert for product state
        q0len = len(self._qubit0)
        q1len = len(self._qubit1)
        # empty contingency table with right dimensions
        cont_table = np.empty((2**q0len,2**q1len))
        cont_table.fill(1)
        for (key, value) in counts.items():
            q0index = int(key[:q0len], 2)
            q1index = int(key[q0len:], 2)
            cont_table[q1index][q0index] = value

        chisq, pval, dof, expctd = chi2_contingency(cont_table)
        if pval <= self._pcrit:
            passed = False
        else:
            passed = True
        return (chisq, pval, passed)


def assertproduct(self, pcrit, qubit0, cbit0, qubit1, cbit1):
    """Create product assertion

    Args:
        expval: integer of 0's and 1's
        pcrit: critical p-value for the hypothesis test
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
    theClone = self.copy("breakpoint"+"_"+AssertManager.breakpoint_name())
    assertion = AssertProduct(pcrit, qubit0, cbit0, qubit1, cbit1) 
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    AssertManager.StatOutputs[theClone.name] = {"type":"Product", "qubit0":assertion._qubit0, \
        "cbit0":assertion._cbit0, "qubit1":assertion._qubit1, "cbit1":assertion._cbit1, \
        "qubit":assertion._qubit, "cbit":assertion._cbit}
    assertion._qubit0 = [assertion._qubit0]
    return theClone

QuantumCircuit.assertproduct = assertproduct
