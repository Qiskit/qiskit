# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test the quantum operators  modules."""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import process_fidelity, Pauli

from ..common import QiskitTestCase


class TestOperators(QiskitTestCase):
    """Tests for qi.py"""

    def test_process_fidelity(self):

        Unitary1 = Pauli(label='XI').to_matrix()
        Unitary2 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
        process_fidelity(Unitary1, Unitary2)
        self.assertAlmostEqual(process_fidelity(Unitary1, Unitary2), 1.0, places=7)
        theta = 0.2
        Unitary1 = expm(-1j*theta*Pauli(label='X').to_matrix()/2)
        Unitary2 = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                             [-1j*np.sin(theta/2), np.cos(theta/2)]])
        self.assertAlmostEqual(process_fidelity(Unitary1, Unitary2), 1.0, places=7)


if __name__ == '__main__':
    unittest.main()
