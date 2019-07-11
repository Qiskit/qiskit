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
from qiskit.assertions.assertmanager import AssertManager
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare

class AssertSuperposition(Asserts):
    """
        Assertion of superposition states and quantum measurement
        in the computational basis.
    """
    def __init__(self, pcrit, qubit, cbit):
        super().__init__()
        self._type = "Superposition"
        self._pcrit = pcrit
        self._qubit = AssertManager.syntax4measure(qubit)
        self._cbit = AssertManager.syntax4measure(cbit)

    def stat_test(self, counts):
        vals_list = list(counts.values())
        numzeros = 2**len(list(counts)[0]) - len(counts)
        vals_list.extend([0]*numzeros)
        chisq, pval = chisquare(vals_list)
        if pval <= self._pcrit or chisq == 0: #math.isnan(pval):
            passed = False
        else:
            passed = True
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
    theClone = self.copy("breakpoint"+"_"+AssertManager.breakpoint_name())
    assertion = AssertSuperposition(pcrit, qubit, cbit)
    theClone.append(assertion, [assertion._qubit], [assertion._cbit])
    AssertManager.StatOutputs[theClone.name] = {"type":"Superposition", \
        "qubit":assertion._qubit, "cbit":assertion._cbit}
    return theClone

QuantumCircuit.assertsuperposition = assertsuperposition
