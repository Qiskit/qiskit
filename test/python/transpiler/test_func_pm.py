# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Size function pass manager"""

import unittest

from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.quantum_info.random import random_unitary
from qiskit.transpiler.preset_passmanagers.level0 import level_0_pass_manager
from qiskit.test.mock import FakeTenerife
from qiskit.test import QiskitTestCase


class TestFuncPassManager(QiskitTestCase):
    """ Tests for Depth analysis methods. """

    def test_simple(self):
        """ Test if func based opt0 equal buitin opt0"""
        N = 5   # pylint: disable=invalid-name
        NQ = N  # pylint: disable=invalid-name
        depth = int(N/2)
        qr = QuantumRegister(N)
        cr = ClassicalRegister(N)
        circ = QuantumCircuit(qr, cr)
        offset = 1
        circ.h(range(NQ))
        for j in range(depth):
            for i in range(int(NQ / 2)):
                k = i * 2 + offset + j % 2
                circ.cx(qr[k % NQ], qr[(k+1) % NQ])
            for i in range(NQ):
                rand_unitary = random_unitary(2)
                circ.append(rand_unitary, [qr[i]])
        circ.measure(range(NQ), range(NQ))

        func_circ = transpile(circ, FakeTenerife(),
                              pass_manager=level_0_pass_manager,
                              seed_transpiler=1)

        opt0_circ = transpile(circ, FakeTenerife(),
                              optimization_level=0,
                              seed_transpiler=1)

        self.assertEqual(func_circ, opt0_circ)


if __name__ == '__main__':
    unittest.main()
