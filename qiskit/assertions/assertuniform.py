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
Assertion of uniform states.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare

class AssertUniform(Asserts):
    """
        Assertion of uniform states and quantum measurement
        in the computational basis.
    """
    def __init__(self, qubit, cbit, pcrit, negate):
        super().__init__(self.syntax4measure(qubit), self.syntax4measure(cbit), pcrit, negate)

    def stat_test(self, counts):
        # chi-squared statistical test to assert
        # uniform superposition of all possible states
        vals_list = list(counts.values())
        numzeros = 2**len(list(counts)[0]) - len(counts)
        vals_list.extend([0]*numzeros)
        chisq, pval = chisquare(vals_list)
        if pval >= self._pcrit:
            passed = True
        else:
            passed = False
        return (chisq, pval, passed)

def assert_uniform(self, qubit, cbit, pcrit=0.05):
    """Create uniform assertion

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
    theClone = self.copy("breakpoint"+"_"+Asserts.breakpoint_name())
    assertion = AssertUniform(qubit, cbit, pcrit, False)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    # AssertManager.StatOutputs[theClone.name] = {"type":"Uniform", \
        # "qubit":assertion._qubit, "cbit":assertion._cbit}
    return theClone

QuantumCircuit.assert_uniform = assert_uniform

def assert_not_uniform(self, qubit, cbit, pcrit=0.05):
    theClone = self.copy("breakpoint"+"_"+Asserts.breakpoint_name())
    assertion = AssertUniform(qubit, cbit, pcrit, True)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    # AssertManager.StatOutputs[theClone.name]["type"] = "Not Uniform"
    return theClone

QuantumCircuit.assert_not_uniform = assert_not_uniform
