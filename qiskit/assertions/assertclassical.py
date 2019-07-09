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
    def __init__(self, qubit, cbit, expval, pcrit):
        super().__init__()
        self._type = "Classical"
        self._qubit = qubit
        self._cbit = cbit
        self._pcrit = pcrit
        self._expval = expval

    def stat_test(self, counts):
        print("counts = ")
        print(counts)
        vals_list = list(counts.values())
        numzeros = 2**len(list(counts)[0]) - len(counts)
        vals_list.extend([0]*numzeros)
        # numshots = sum(list(counts.values()))
        
        exp_list = [1]*len(vals_list)
        try:
            index = list(map(int, counts.keys())).index(self._expval)
        except ValueError:
            index = -1
        exp_list[index] = 2**16
        print("vals_list = ")
        print(vals_list)
        print("exp_list = ")
        print(exp_list)
        vals_list = vals_list / np.sum(vals_list)
        exp_list = exp_list / np.sum(exp_list)
        chisq, pval = chisquare(vals_list, f_exp = exp_list, ddof=1)
        if len(list(counts.keys())[0]) == 1:
            pval = vals_list[index]
            passed = True if pval >= 1 - self._pcrit else False
        else:
            passed = True if pval >= self._pcrit else False
        print("chisq, pval = ")
        print(chisq, pval)
        return (chisq, pval, passed)


def assertclassical(self, expval, pcrit, qubit, cbit):
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
    AssertManager.StatOutputs[theClone.name] = {"type":"Classical","expval":expval}
    theClone.append(AssertClassical(qubit, cbit, expval, pcrit), [qubit], [cbit])
    return theClone

QuantumCircuit.assertclassical = assertclassical
