# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
from scipy.stats import chisquare
from qiskit.assertions.asserts import Asserts
from qiskit.circuit.quantumcircuit import QuantumCircuit


class AssertUniform(Asserts):
    """
        A measurement instruction that additionally performs statistical tests on the measurement
        outcomes to assert whether the state is a uniform superposition state or not.
    """
    def __init__(self, qubit, cbit, pcrit, negate):
        """
        Constructor for AssertUniform

        Args:
            qubit(QuantumRegister or list): quantum register
            cbit(ClassicalRegister or list): classical register
            pcrit(float): the critical p-value
            negate(bool): True if assertion passed is negation of statistical test passed
        """
        type_str = "Not Uniform" if negate else "Uniform"
        super().__init__(self._syntax_for_measure(qubit), self._syntax_for_measure(cbit),
                         pcrit, negate, type_str)

    def stat_test(self, counts):
        """
        Performs a chi-squared statistical test on the experimental outcomes.  Internally, compares
        a normalized table of experimental counts to the scipy.stats.chisquare default, for which
        all outcomes are equally likely.

        Args:
            counts(dictionary): result.get_counts(experiment)

        Returns:
            tuple: tuple containing:

                chisq(float): the chi-square value

                pval(float): the p-value

                passed(Boolean): if the test passed
        """

        vals_list = list(counts.values())

        # account for bitstring with zero counts
        numqubits = len(list(counts)[0])
        numzeros = 2 ** numqubits - len(counts)
        vals_list.extend([0] * numzeros)

        chisq, pval = chisquare(vals_list)
        passed = bool(pval >= self._pcrit)
        return (chisq, pval, passed)


def get_breakpoint_uniform(self, qubit, cbit, pcrit=0.05):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertUniform instruction to its end.  If the statistical test passes, the
    assertion passes; if the test fails, the assertion fails.

    Args:
        qubit (QuantumRegister or list): quantum register
        cbit (ClassicalRegister or list): classical register
        pcrit (float): critical p-value for the hypothesis test

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    clone = self.copy(Asserts._new_breakpoint_name())
    assertion = AssertUniform(qubit, cbit, pcrit, False)
    clone.append(assertion, [assertion._qubit], [assertion._cbit])
    return clone


QuantumCircuit.get_breakpoint_uniform = get_breakpoint_uniform


def get_breakpoint_not_uniform(self, qubit, cbit, pcrit=0.05):
    """
    Creates a breakpoint, which is a renamed deep copy of the QuantumCircuit, and creates and
    appends an AssertUniform instruction to its end.  If the statistical test passes, the
    assertion fails; if the test fails, the assertion passes.

    Args:
        qubit (QuantumRegister or list): quantum register
        cbit (ClassicalRegister or list): classical register
        pcrit (float): critical p-value for the hypothesis test

    Returns:
        QuantumCircuit: copy of quantum circuit at the assert point
    """
    clone = self.copy(Asserts._new_breakpoint_name())
    assertion = AssertUniform(qubit, cbit, pcrit, True)
    clone.append(assertion, [assertion._qubit], [assertion._cbit])
    return clone


QuantumCircuit.get_breakpoint_not_uniform = get_breakpoint_not_uniform
