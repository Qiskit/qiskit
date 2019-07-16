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
from qiskit.assertions.assertmanager import AssertManager
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare

class AssertClassical(Asserts):
    """
        Assertion of classical states and quantum measurement
        in the computational basis.
    """
    def __init__(self, expval, pcrit, qubit, cbit):
        super().__init__()
        self._expval = expval if isinstance(expval, int) else int(expval, 2)
        self._pcrit = pcrit
        self._qubit = AssertManager.syntax4measure(qubit)
        self._cbit = AssertManager.syntax4measure(cbit)
        if not self._expval in range(0, 2**len(self._cbit)):
            raise QiskitError("AssertClassical expected value %d not in range for %d cbits" % (self._expval, len(self._cbit)))

    def stat_test(self, counts):
        # chi-squared statistical test for classical assertions
        vals_list = list(counts.values())
        numqubits = len(list(counts)[0])
        numzeros = 2**numqubits - len(counts)
        vals_list.extend([0]*numzeros)
        
        exp_list = [1]*len(vals_list)
        try:
            index = list(map(lambda x: int(x, 2), counts.keys())).index(self._expval)
        except ValueError:
            index = -1
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


def assert_classical(self, expval, pcrit, qubit, cbit):
    """Create classical assertion

    Args:
        expval: integer of 0's and 1's
        pcrit: critical p-value for the hypothesis test
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.QuantumCircuit: copy of quantum circuit at the assert point.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    theClone = self.copy("breakpoint"+"_"+AssertManager.breakpoint_name())
    assertion = AssertClassical(expval, pcrit, qubit, cbit)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    AssertManager.StatOutputs[theClone.name] = {"type":"Classical", "expval":expval, \
        "qubit":assertion._qubit, "cbit":assertion._cbit}
    return theClone

QuantumCircuit.assert_classical = assert_classical

def assert_not_classical(self, expval, pcrit, qubit, cbit):
    theClone = assert_classical(expval, pcrit, qubit, cbit)
    AssertManager.StatOutputs[theClone.name]["type"] = "Not Classical"
    return theClone

QuantumCircuit.assert_not_classical = assert_not_classical
