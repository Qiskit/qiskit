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
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chi2_contingency

class AssertProduct(Asserts):
    """
        Assertion of product states and quantum measurement
        in the computational basis.
    """
    def __init__(self, qubit0, cbit0, qubit1, cbit1, pcrit, negate):
        self._qubit0 = self.syntax4measure(qubit0)
        self._cbit0 = self.syntax4measure(cbit0)
        self._qubit1 = self.syntax4measure(qubit1)
        self._cbit1 = self.syntax4measure(cbit1)
        super().__init__(self._qubit0 + self._qubit1, self._cbit0 + self._cbit1, pcrit, negate)

    def stat_test(self, counts):
        """Performs a chi-squared contingency test on the experimental outcomes.
        Internally, constructs a contingency table from the experimental counts.
        

        Args:
            counts: the counts dictionary from result.get_counts()
        
        Returns:
            chisq: the chi-square value
            pval: the p-value
            passed: a boolean that is True iff the assertion passed
        """
        q0len = len(self._qubit0)
        q1len = len(self._qubit1)
        cont_table = np.empty((2**q0len,2**q1len))
        cont_table.fill(1)
        for (key, value) in counts.items():
            q0index = int(key[:q0len], 2)
            q1index = int(key[q0len:], 2)
            cont_table[q0index][q1index] = value

        chisq, pval, dof, expctd = chi2_contingency(cont_table)
        if pval >= self._pcrit:
            passed = True
        else:
            passed = False
        return (chisq, pval, passed)

def assert_product(self, qubit0, cbit0, qubit1, cbit1, pcrit=0.05):
    """Create product assertion

    Args:
        pcrit: critical p-value for the hypothesis test
        qubit0 (QuantumRegister|list|tuple): quantum register
        cbit0 (ClassicalRegister|list|tuple): classical register
        qubit1 (QuantumRegister|list|tuple): quantum register
        cbit1 (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.QuantumCircuit: copy of quantum circuit at the assert point.
    """
    theClone = self.copy(Asserts.breakpoint_name())
    assertion = AssertProduct(qubit0, cbit0, qubit1, cbit1, pcrit, False)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    return theClone

QuantumCircuit.assert_product = assert_product

def assert_not_product(self, qubit0, cbit0, qubit1, cbit1, pcrit=0.05):
    theClone = self.copy(Asserts.breakpoint_name())
    assertion = AssertProduct(qubit0, cbit0, qubit1, cbit1, pcrit, True)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    return theClone

QuantumCircuit.assert_not_product = assert_not_product
