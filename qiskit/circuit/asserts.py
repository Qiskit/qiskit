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
Quantum measurement in the computational basis.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from random import randint
from scipy.stats import chisquare


class Asserts(Measure):
    wantcsv = True
    StatOutputs = {}
    """Quantum measurement in the computational basis."""
    def __init__(self):
        """Create new measurement instruction."""
        super().__init__()


def stat_test(experiments, results):
#need to fix comments on functions
    """Create classical assertion

    Args: #fix these!
        expval: integer
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.QuantumCircuit: copy of quantum circuit at the assert point.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    for exp in experiments:
        exp_results = results.get_counts(exp)
        print(list(exp_results.values()))        
        exptype = Asserts.StatOutputs[exp.name]["type"]
        #print(exptype)
        #
        #splitting by type here, we can change how this is done later
        if exptype == "Classical":
            c, p = (chisquare(list(exp_results.values()))) #placeholder, this should be replaced by a call to assertclassical.stat_test
        elif exptype == "Superposition":
            c, p = (chisquare(list(exp_results.values()))) #this is what should be implemented here, but by a call to assertsuperposition.stat_test
        elif exptype == "Product":
            c, p = (chisquare(list(exp_results.values())))  #placeholder, this should be replaced by a call to assertclassical.stat_test
        else: print("Error in asserts.stat_test: experiment doesn't have a recorded type")
        #
        print(c, p)
        Asserts.StatOutputs[exp.name]["chisq"] = c
        Asserts.StatOutputs[exp.name]["p"] = p
        #now the dict StatOutputs should map each breakpoint.name to another dictionary containing type, chisq, p, as well as other inputs like expval
    return Asserts.StatOutputs



#def output_csv():
    #return something
