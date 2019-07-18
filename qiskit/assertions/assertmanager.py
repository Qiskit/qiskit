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
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.exceptions import QiskitError
from datetime import datetime

class AssertManager():
    """An AssertManager object manages all assertions in the experiment and executes them."""
    StatOutputs = {}

    def breakpoint_name():
        return "breakpoint_" + datetime.now().isoformat()

    def syntax4measure(bit):
    # support for all known measure syntaxes
        if isinstance(bit,(list, Register)):
            return bit
        elif isinstance(bit,(range, tuple)):
            return list(bit)
        else: #if single bit
            return [bit]

    def cbits2idxs(cbits, exp):
    # gives index wrt counts object for clbits
        if isinstance(cbits[0], int): # syntax 1
            return cbits
        elif isinstance(cbits[0], Clbit): # syntax 2
            idxs = [exp.clbits.index(cbit) for cbit in cbits]
            return idxs
        elif isinstance(cbits, ClassicalRegister): # syntax 3
            idxfirst = exp.clbits.index(cbits[0])
            idxlast = exp.clbits.index(cbits[-1])
            return range(idxfirst, idxlast+1)

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
        if not isinstance(experiments, list):
            experiments = [experiments]
        for exp in experiments:
            exp_counts = results.get_counts(exp)
            assertion = exp.data[-1][0]
            cbits = assertion._cbit
            cbits = AssertManager.cbits2idxs(cbits, exp)

            new_counts = {}
            for (key, value) in exp_counts.items():
                newkey = key.replace(' ', '')
                newkey = ''.join([newkey[-1*(cbit+1)] for cbit in cbits][::-1])
                new_counts.setdefault(newkey, 0)
                new_counts[newkey] += value
            exp_counts = new_counts

            chisq, pval, passed = assertion.stat_test(exp_counts)
            AssertManager.StatOutputs[exp.name]["expval"] = assertion._expval
            AssertManager.StatOutputs[exp.name]["pcrit"] = assertion._pcrit
            AssertManager.StatOutputs[exp.name]["chisq"] = chisq
            AssertManager.StatOutputs[exp.name]["pval"] = pval
            AssertManager.StatOutputs[exp.name]["passed"] = passed if AssertManager.StatOutputs[exp.name]["type"][0:3] != "Not" else not passed
            AssertManager.StatOutputs[exp.name]["counts"] = exp_counts
            # now the dict StatOutputs should map each breakpoint.name to another dictionary containing type, chisq, p, as well as other inputs like expvalue:
            stat_output = AssertManager.StatOutputs[exp.name]
            # prints output message:
            print(('PASSED: ' if stat_output['passed'] else 'FAILED: ') + stat_output['type'] + " Assertion on qubits " + \
                ((str(stat_output['qubit'])+" measured to cbits "+str(stat_output['cbit'])) if not 'qubit0' in stat_output \
                else (str(stat_output['qubit0'])+str(stat_output['qubit1']))+" measured to cbits "+str(stat_output['cbit0'])+str(stat_output['cbit1'])))
        return AssertManager.StatOutputs
