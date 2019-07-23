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
Abstract class and all derived classes for making statistical assertions.
"""
import abc
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare
from datetime import datetime

class Asserts(Measure):
    """Superclass for all the asserts, subclass of Measure"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, qubit, cbit, pcrit, negate):
        super().__init__()
        self._qubit = qubit
        self._cbit = cbit
        self._pcrit = pcrit
        self._negate = negate
        self._expval = None

    def breakpoint_name():
        return "breakpoint_" + datetime.now().isoformat()

    def syntax4measure(self, bit):
    # support for all known measure syntaxes
        if isinstance(bit,(list, Register)):
            return bit
        elif isinstance(bit,(range, tuple)):
            return list(bit)
        else: #if single bit
            return [bit]

    def clbits2idxs(cbits, exp):
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

    @abc.abstractmethod
    def stat_test(self, counts):
        """Performs a statistical test on the experimental outcomes

        Args:
            counts (results.get_counts): counts of result of an experiment

        Returns:
            tuple containing the chisquare, p-value, and passed boolean of the assertion
        """
        return
