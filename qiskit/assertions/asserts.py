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
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from scipy.stats import chisquare

class Asserts(Measure):
    """Superclass for all the asserts, subclass of Measure"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()
        self._qubit = None
        self._cbit = None
        self._pcrit = None
        self._expval = None

    @abc.abstractmethod
    def stat_test(self, counts):
        """Performs a statistical test on the experimental outcomes

        Args:
            counts (results.get_counts): counts of result of an experiment

        Returns:
            tuple containing the chisquare, p-value, and success boolean of the assertion
        """
        return
