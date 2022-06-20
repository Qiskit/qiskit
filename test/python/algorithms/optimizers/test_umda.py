# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the UMDA optimizer."""

from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from qiskit.algorithms.optimizers.umda import UMDA


class TestUMDA(QiskitAlgorithmsTestCase):
    """Tests for the UMDA optimizer."""

    def test_get_set(self):
        """Test if getters and setters work as expected"""
        umda = UMDA(maxiter=1, size_gen=20)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.dead_iter = 10
        umda.max_iter = 100

        assert umda.disp is True
        assert umda.size_gen == 30
        assert umda.alpha == 0.6
        assert umda.dead_iter == 10
        assert umda.max_iter == 100

    def test_settings(self):
        """Test if the settings display works well"""
        umda = UMDA(maxiter=1, size_gen=20)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.dead_iter = 10
        umda.max_iter = 100

        set_ = {
            "max_iter": 100,
            "alpha": 0.6,
            "dead_iter": 10,
            "size_gen": 30,
        }

        assert umda.settings == set_
