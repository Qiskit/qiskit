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
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.exceptions import QiskitError
from datetime import datetime

class AssertManager():
    """An AssertManager object manages all assertions in the experiment and executes them."""
    StatOutputs = {}

    def breakpoint_name():
        return datetime.now().isoformat()

    def clbits2idxs(cbits, exp):
        if isinstance(cbits, ClassicalRegister): # syntax 3
            idxfirst = exp.clbits.index(cbits[0])
            idxlast = exp.clbits.index(cbits[-1])
            return range(idxfirst, idxlast+1)
        elif isinstance(cbits, int) or isinstance(cbits, Clbit):
            cbits = [cbits]
        if isinstance(cbits[0], int): # syntax 1
            return cbits
        elif isinstance(cbits[0], Clbit): # syntax 2
            idxs = [exp.clbits.index(cbit) for cbit in cbits]
            return idxs

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
        if isinstance(experiments, QuantumCircuit):
            experiments = [experiments]
        for exp in experiments:
            print("exp.data")
            print(exp.data)
            exp_counts = results.get_counts(exp)
            print(exp_counts)
            assertion = exp.data[-1][0]
            cbits = assertion._cbit
            print(assertion)
            exp_type = assertion.get_type()
            cbits = AssertManager.clbits2idxs(cbits, exp)
            print("cbits")
            print(cbits)

            new_counts = {}
            for (key, value) in exp_counts.items():
                newkey = key.replace(' ', '')
                newkey = ''.join([newkey[-1*(cbit+1)] for cbit in cbits][::-1])
                new_counts.setdefault(newkey, 0)
                new_counts[newkey] += value
            exp_counts = new_counts


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
