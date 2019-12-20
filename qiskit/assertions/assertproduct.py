# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
from scipy.stats import chi2_contingency
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumcircuit import Register


class AssertProduct(Asserts):
    """
        A measurement instruction that additionally performs statistical tests on the measurement
        outcomes to assert whether the state is a product state or not.
    """
    def __init__(self, qubit0, cbit0, qubit1, cbit1, pcrit, negate):
        """
        Constructor for AssertProduct

        Args:
            qubit0(QuantumRegister or list): quantum register
            cbit0(ClassicalRegister or list): classical register
            qubit1(QuantumRegister or list): quantum register
            cbit1(ClassicalRegister or list): classical register
            pcrit(float): the critical p-value
            negate(bool): True if assertion passed is negation of statistical test passed
        """
        self._qubit0 = self._syntax_for_measure(qubit0)
        self._cbit0 = self._syntax_for_measure(cbit0)
        self._qubit1 = self._syntax_for_measure(qubit1)
        self._cbit1 = self._syntax_for_measure(cbit1)
        type_str = "Not Product" if negate else "Product"
        qubit0 = list(self._qubit0) if isinstance(self._qubit0, Register) else self._qubit0
        qubit1 = list(self._qubit1) if isinstance(self._qubit1, Register) else self._qubit1
        cbit0 = list(self._cbit0) if isinstance(self._cbit0, Register) else self._cbit0
        cbit1 = list(self._cbit1) if isinstance(self._cbit1, Register) else self._cbit1
        super().__init__(qubit0 + qubit1, cbit0 + cbit1, pcrit, negate, type_str)

    def stat_test(self, counts):
        """
        Performs a chi-squared contingency test on the experimental outcomes.
        Internally, constructs a contingency table from the experimental counts.


        Args:
            counts(dictionary): result.get_counts(experiment)

        Returns:
            tuple: tuple containing:

                chisq(float): the chi-square value

                pval(float): the p-value

                passed(bool): if the test passed
        """
        q0len = len(self._qubit0)
        q1len = len(self._qubit1)
        cont_table = np.ones((2 ** q0len, 2 ** q1len))

        for (key, value) in counts.items():
            key_rev = key[::-1]
            q0index = int(key_rev[:q0len], 2)
            q1index = int(key_rev[q0len:], 2)
            cont_table[q0index][q1index] = value

        chisq, pval, _, _ = chi2_contingency(cont_table)
        passed = bool(pval >= self._pcrit)
        return (chisq, pval, passed)


def get_breakpoint_product(self, qubit0, cbit0, qubit1, cbit1, pcrit=0.05):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertProduct instruction to its end.  It tests whether qubit0 and qubit1, measured
    to cbit0 and cbit1, have any entanglement between them.  If the statistical test passes, the
    assertion passes; if the test fails, the assertion fails.

    Args:
        qubit0(QuantumRegister or list): quantum register
        cbit0(ClassicalRegister or list): classical register
        qubit1(QuantumRegister or list): quantum register
        cbit1(ClassicalRegister or list): classical register
        pcrit(float): critical p-value for the hypothesis test

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    clone = self.copy(Asserts._new_breakpoint_name())
    assertion = AssertProduct(qubit0, cbit0, qubit1, cbit1, pcrit, False)
    clone.append(assertion, [assertion._qubit], [assertion._cbit])
    return clone


QuantumCircuit.get_breakpoint_product = get_breakpoint_product


def get_breakpoint_not_product(self, qubit0, cbit0, qubit1, cbit1, pcrit=0.05):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertProduct instruction to its end.  It tests whether qubit0 and qubit1, measured
    to cbit0 and cbit1, have any entanglement between them.  If the statistical test passes, the
    assertion fails; if the test fails, the assertion passes.

    Args:
        qubit0(QuantumRegister or list): quantum register
        cbit0(ClassicalRegister or list): classical register
        qubit1(QuantumRegister or list): quantum register
        cbit1(ClassicalRegister or list): classical register
        pcrit(float): critical p-value for the hypothesis test

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    clone = self.copy(Asserts._new_breakpoint_name())
    assertion = AssertProduct(qubit0, cbit0, qubit1, cbit1, pcrit, True)
    clone.append(assertion, [assertion._qubit], [assertion._cbit])
    return clone


QuantumCircuit.get_breakpoint_not_product = get_breakpoint_not_product
