# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for resets in the transpiler"""

import unittest

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase


class TestResetsTranspiler(QiskitTestCase):
    """Tests resets and transpilation"""

    def test_init_resets_kept_preset_passmanagers(self):
        """Test initial resets kept at all preset transpilation levels"""
        N = 5
        qc = QuantumCircuit(N)
        qc.reset(range(N))

        for level in range(4):
            num_resets = transpile(qc, optimization_level=level).count_ops()["reset"]
            self.assertEqual(num_resets, N)
