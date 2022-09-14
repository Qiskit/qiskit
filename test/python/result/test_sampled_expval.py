# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit.quantum_info.analysis"""

import unittest

from qiskit.result import Counts, QuasiDistribution, ProbDistribution, sampled_expectation_value
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow import PauliOp, PauliSumOp
from qiskit.test import QiskitTestCase


PROBS = {'1000': 0.0022,
         '1001': 0.0045,
         '1110': 0.0081,
         '0001': 0.0036,
         '0010': 0.0319,
         '0101': 0.001,
         '1100': 0.0008,
         '1010': 0.0009,
         '1111': 0.3951,
         '0011': 0.0007,
         '0111': 0.01,
         '0000': 0.4666,
         '1101': 0.0355,
         '1011': 0.0211,
         '0110': 0.0081,
         '0100': 0.0099
         }


class TestSampledExpval(QiskitTestCase):
    """Test sampled expectation values"""

    def test_simple(self):
        """Test that basic exp values work"""

        DIST2 = {'00': 0.5, '11': 0.5}
        DIST3 = {'000': 0.5, '111': 0.5}
        # ZZ even GHZ is 1.0
        self.assertAlmostEqual(sampled_expectation_value(DIST2, 'ZZ'), 1.0)
        # ZZ odd GHZ is 0.0
        self.assertAlmostEqual(sampled_expectation_value(DIST3, 'ZZZ'), 0.0)
        # All id ops goes to 1.0
        self.assertAlmostEqual(sampled_expectation_value(DIST3, 'III'), 1.0)
        # flipping one to I makes even GHZ 0.0
        self.assertAlmostEqual(sampled_expectation_value(DIST2, 'IZ'), 0.0)
        self.assertAlmostEqual(sampled_expectation_value(DIST2, 'ZI'), 0.0)
        # Generic Z on PROBS
        self.assertAlmostEqual(sampled_expectation_value(PROBS, 'ZZZZ'), 0.7554)
    
    
    def test_same(self):
        """Test that all operators agree with each other for dict input"""
        ANS = 0.9356
        counts = {'001': 67, '110': 113, '100': 83, '011': 205,
                  '111': 4535, '101': 100, '010': 42, '000': 4855}
        OPER = 'IZZ'

        exp1 = sampled_expectation_value(counts, OPER)
        self.assertAlmostEqual(exp1, ANS)

        exp2 = sampled_expectation_value(counts, Pauli(OPER))
        self.assertAlmostEqual(exp2, ANS)
        
        exp3 = sampled_expectation_value(counts, PauliOp(Pauli(OPER)))
        self.assertAlmostEqual(exp3, ANS)

        exp4 = sampled_expectation_value(counts, PauliSumOp.from_list([[OPER, 1]]))
        self.assertAlmostEqual(exp4, ANS)
        
        exp5 = sampled_expectation_value(counts, SparsePauliOp.from_list([[OPER, 1]]))
        self.assertAlmostEqual(exp5, ANS)


    def test_asym_ops(self):
        """Test that asymmetric exp values work"""
        DIST = QuasiDistribution(PROBS)
        self.assertAlmostEqual(sampled_expectation_value(DIST, '0III'), 0.5318)
        self.assertAlmostEqual(sampled_expectation_value(DIST, 'III0'), 0.5285)
        self.assertAlmostEqual(sampled_expectation_value(DIST, '1011'), 0.0211)


if __name__ == "__main__":
    unittest.main(verbosity=2)
