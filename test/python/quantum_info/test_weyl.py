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
# pylint: disable=invalid-name

"""Tests for Weyl coorindate routines."""

import unittest

import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis.weyl import weyl_coordinates, local_equivalence
from qiskit.quantum_info.synthesis.local_invariance import two_qubit_local_invariants


class TestWeyl(QiskitTestCase):
    """Test Weyl coordinate routines"""

    def test_weyl_coordinates(self):
        """Randomly check Weyl coordinates math local invariants.
        """
        for _ in range(10):
            U = random_unitary(4).data
            weyl = weyl_coordinates(U)
            local_equiv = local_equivalence(weyl)
            local = two_qubit_local_invariants(U)
            self.assertTrue(np.allclose(local, local_equiv))


if __name__ == '__main__':
    unittest.main()
