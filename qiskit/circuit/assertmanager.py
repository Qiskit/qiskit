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
import time
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError

class AssertManager():
    """An AssertManager object manages all assertions in the experiment and executes them."""
    StatOutputs = {}

    def breakpoint_name():
        tl = time.localtime()
        ta = time.asctime()
        numzeros = 2 - len(str(tl.tm_mon))
        sy = str(tl.tm_year)
        sm = str(tl.tm_mon)
        timestring = sy+"_"+"0"*numzeros+sm+ta.replace(" ","_").replace(":","-")[7:-5]
        if timestring in StatOutputs:
            if timestring[-3] == "-":
                return timestring+"_1"
            else:
                index = timestring.find("-")+7
                return timestring[:index]+str(int(timestring[index:])+1)

    def stat_collect(experiments, results):
        """Calculate and collect results of statistical tests for each experiment
    
        Args:
            experiments (list[QuantumCircuit]): a list of all breakpoints
            results (list[Results]): a list of the results of all the experiments

        Returns:
            passed (list[bool]): a list of booleans that is true if each test passed

        Raises:
            ?: if experiments and results are not the same length
        """
        for exp in experiments:
            exp_counts = results.get_counts(exp)
            print(list(exp_counts.values()))
            assertion = exp.data[-1][0]
            print(assertion)
            exp_type = assertion.get_type()
            print(exp_type)
            chisq, pval, passed = assertion.stat_test(exp_counts)
            print(chisq, pval, passed)
            AssertManager.StatOutputs[exp.name]["type"] = assertion.get_type()
            AssertManager.StatOutputs[exp.name]["expval"] = assertion.get_expval()
            AssertManager.StatOutputs[exp.name]["pcrit"] = assertion.get_pcrit()
            AssertManager.StatOutputs[exp.name]["chisq"] = chisq
            AssertManager.StatOutputs[exp.name]["pval"] = pval
            AssertManager.StatOutputs[exp.name]["passed"] = passed
            #now the dict StatOutputs should map each breakpoint.name to another dictionary containing type, chisq, p, as well as other inputs like expval
        return AssertManager.StatOutputs

    #def output_csv():
        #return something
