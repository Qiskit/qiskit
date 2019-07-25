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
Assertion of classical states.
"""
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare

class AssertClassical(Asserts):
    """
        A measurement instruction that additionally performs statistical tests on the measurement outcomes
	to assert whether the state is a classical state or not.
    """
    def __init__(self, qubit, cbit, pcrit, expval, negate):
        super().__init__(self.syntax4measure(qubit), self.syntax4measure(cbit), pcrit, negate)
        self._expval = expval if expval==None or isinstance(expval, int) else int(expval, 2)
        if expval!=None and not self._expval in range(0, 2**len(self._cbit)):
            raise QiskitError("AssertClassical expected value %d not in range for %d cbits" % (self._expval, len(self._cbit)))

    def stat_test(self, counts):
        """
        Performs a chi-squared statistical test on the experimental outcomes.  Internally, it
        builds a table of all 1's except is 2^16 for the expected value, normalizes it, and
        compares to a normalized table of experimental counts. 
        
        Args:
            counts(dictionary): result.get_counts(experiment)

        Returns:
            tuple: tuple containing:
            
                chisq(float): the chi-squared value
                pval(float): the p-value
                passed(Boolean): if the test passed
        """

        vals_list = list(counts.values())
        if self._expval == None:
            index = np.argmax(vals_list)
        else:
            try:
                index = list(map(lambda x: int(x, 2), counts.keys())).index(self._expval)
            except ValueError:
                index = -1
        numqubits = len(list(counts)[0])
        numzeros = 2**numqubits - len(counts)
        vals_list.extend([0]*numzeros)
        exp_list = [1]*len(vals_list)
        exp_list[index] = 2**16
        vals_list = vals_list / np.sum(vals_list)
        exp_list = exp_list / np.sum(exp_list)
        chisq, pval = chisquare(vals_list, f_exp = exp_list, ddof=1)
        if len(list(counts.keys())[0]) == 1:
            pval = vals_list[index]
            passed = True if pval >= 1 - self._pcrit else False
        else:
            passed = True if pval >= self._pcrit else False
        return (chisq, pval, passed)


def get_breakpoint_classical(self, qubit, cbit, pcrit=0.05, expval=None):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertClassical instruction to its end.  If no expected value is specified, the
    default behavior is that the statistical test assumes the mode of the measurement results
    is the expected value.  If the statistical test passes, the assertion passes; if the test
    fails, the assertion fails. 

    Args:
        expval (integer|string): integer in base 10, or a string of 0's and 1's
        pcrit (float): critical p-value for the hypothesis test
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    theClone = self.copy(Asserts.breakpoint_name())
    assertion = AssertClassical(qubit, cbit, pcrit, expval, False)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    return theClone

QuantumCircuit.get_breakpoint_classical = get_breakpoint_classical

def get_breakpoint_not_classical(self, qubit, cbit, pcrit=0.05, expval=None):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertClassical instruction to its end.  If no expected value is specified, the
    default behavior is that the statistical test assumes the mode of the measurement results
    is the expected value.  If the statistical test passes, the assertion fails; if the test
    fails, the assertion passes.

    Args:
        expval (integer|string): integer in base 10, or a string of 0's and 1's
        pcrit (float): critical p-value for the hypothesis test
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    theClone = self.copy(Asserts.breakpoint_name())
    assertion = AssertClassical(qubit, cbit, pcrit, expval, True)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    return theClone

QuantumCircuit.get_breakpoint_not_classical = get_breakpoint_not_classical
