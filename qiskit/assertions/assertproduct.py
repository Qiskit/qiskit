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
    def __init__(self, qubits, cbits, expval, pcrit): #qubit&cbit lists here
        super().__init__()
        self._type = "Product"
        self._qubits = qubits
        self._cbits = cbits
        self._pcrit = pcrit
        self._expval = expval

    def stat_test(self, counts):
        vals_list = list(counts.values())
        numzeros = 2**len(list(counts)[0]) - len(counts)
        vals_list.extend([0]*numzeros)
        # numshots = sum(list(counts.values()))
        
        cont_table = []

        exp_list = [1]*len(vals_list)
        try:
            index = list(map(int, counts.keys())).index(self._expval)
        except ValueError:
            index = -1
        exp_list[index] = 2**16
        
        print("cont_table")
        print(cont_table)
        chisq, pval = (chi2_contingency(cont_table))
        print("chisq, pval = ")
        print(chisq, pval)
        if pval <= self._pcrit:
            passed = True
        else:
            passed = False
        return (chisq, pval, passed)


def assertproduct(self, expval, pcrit, qubit0, cbit0, qubit1, cbit1):
    """Create product assertion

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
    AssertManager.StatOutputs[theClone.name] = {"type":"Product","expval":expval}
    qubitsjoined = []
    for q in qubits:
        qubitsjoined.extend(q)
    cbitsjoined = []
    for c in cbits:
        cbitsjoined.extend(q)
    theClone.append(AssertProduct(qubits, cbits, expval, pcrit), [qubitsjoined], [cbitsjoined])
    return theClone

QuantumCircuit.assertproduct = assertproduct
