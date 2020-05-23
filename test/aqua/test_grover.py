# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" test Grover """

import unittest
import itertools
from test.aqua import QiskitAquaTestCase
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle as LEO, TruthTableOracle as TTO


TESTS = [
    ['p cnf 3 5 \n -1 -2 -3 0 \n 1 -2 3 0 \n 1 2 -3 0 \n 1 -2 -3 0 \n -1 2 3 0',
     ['101', '000', '011'], LEO],
    ['p cnf 2 2 \n 1  0 \n -2  0', ['01'], LEO],
    ['p cnf 2 4 \n 1  0 \n -1 0 \n 2  0 \n -2 0', [], LEO],
    ['a & b & c', ['111'], LEO],
    ['(a ^ b) & a & b', [], LEO],
    ['a & b | c & d', ['0011', '1011', '0111', '1100', '1101', '1110', '1111'], LEO],
    ['1000000000000001', ['0000', '1111'], TTO],
    ['00000000', [], TTO],
    ['0001', ['11'], TTO],
]

MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
SIMULATORS = ['statevector_simulator', 'qasm_simulator']
OPTIMIZATIONS = [True, False]
LAMBDA = [1.44, 8/7]
ROTATION_COUNTS = [
    None,
    [0, 0, 0, 1, 1, 0, 1, 1, 2, 1, 2, 3, 1, 4, 5, 1, 6, 2, 7, 9,
     11, 13, 16, 5, 20, 24, 28, 34, 2, 41, 49, 4, 60]
]


@ddt
class TestGrover(QiskitAquaTestCase):
    """ Grover test """
    @idata(
        [x[0] + list(x[1:]) for x in list(itertools.product(TESTS, MCT_MODES, SIMULATORS,
                                                            OPTIMIZATIONS, LAMBDA,
                                                            ROTATION_COUNTS))]
    )
    @unpack
    def test_grover(self, input_test, sol, oracle_cls, mct_mode,
                    simulator, optimization, lam, rotation_counts):
        """ grover test """
        groundtruth = sol
        oracle = oracle_cls(input_test, optimization=optimization)
        grover = Grover(oracle, incremental=True,
                        lam=lam,
                        rotation_counts=rotation_counts, mct_mode=mct_mode)
        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=1000)

        ret = grover.run(quantum_instance)

        self.log.debug('Ground-truth Solutions: %s.', groundtruth)
        self.log.debug('Top measurement:        %s.', ret['top_measurement'])
        if ret['oracle_evaluation']:
            self.assertIn(ret['top_measurement'], groundtruth)
            self.log.debug('Search Result:          %s.', ret['result'])
        else:
            self.assertEqual(groundtruth, [])
            self.log.debug('Nothing found.')


if __name__ == '__main__':
    unittest.main()
