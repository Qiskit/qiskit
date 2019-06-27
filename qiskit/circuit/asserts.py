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
Superclass for all Assertions.
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
    """Superclass for all the asserts, subclass of Measure"""
    def __init__(self):
        super().__init__()


def stat_test(experiments, results):
#need to fix comments on functions
    """Does statistical tests for each assertion

    Args:
        experiments (QuantumCircuit or list[QuantumCircuit]): Circuit(s) to execute
        results: results object (output.results from output of execute function)

    Returns:
        dictionary containing each breakpoint.name as keys and another dictionary for its values that contains the type of assertion and chisq and p values.

    Raises:
        Error if input experiment hasn't been recorded as an assertion.
    """
    for exp in experiments:
        exp_results = results.get_counts(exp)
        print("list(exp_results.values()) = ")
        print(list(exp_results.values()))        
        exp_type = Asserts.StatOutputs[exp.name]["type"]
        print("exp_type = ")
        print(exp_type)

        #splitting by type here, we can change how this is done later
        #in future we can change to a switch statement in which
        #for each type we call its respective stat_test funtion 
        if exp_type == "Classical":
            res_list = []
            exp_list = []
            numshots = sum(list(exp_results.values()))
            for key, value in exp_results.items():
                res_list.append(value)
                if int(key) == Asserts.StatOutputs[exp.name]["expval"]:
                    exp_list.append(numshots)
                else:
                    exp_list.append(0)
            print("exp_list =")
            print(exp_list)
            print("res_list = ")
            print(res_list)
            c, p = (chisquare(res_list, f_exp = exp_list)) 

        elif exp_type == "Superposition":
            c, p = (chisquare(list(exp_results.values())))

        elif exp_type == "Product":
            c, p = (chisquare(list(exp_results.values()))) 
            #placeholder, this should be replaced by the stat_test for Product

        else: print("Error in asserts.stat_test: experiment doesn't have a recorded type")

        print("c, p =")
        print(c, p)
        Asserts.StatOutputs[exp.name]["chisq"] = c
        Asserts.StatOutputs[exp.name]["p"] = p
        #the dict StatOutputs should map each breakpoint.name to 
        #another dictionary containing type, chisq, p, other inputs like expval
    return Asserts.StatOutputs



#def output_csv():
    #return something
