# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Coupling Graph Test: Directed graph object for representing coupling between qubits."""

import unittest
from qiskit.mapper._coupling import Coupling
from qiskit.mapper._couplingerror import CouplingError
from .common import QiskitTestCase


class TestCoupling(QiskitTestCase):
    """Qiskit Coupling Graph Tests."""

    def test_distance_error(self):
        """Test distance method validation.
        """
        graph = Coupling({0: [1, 2], 1: [2]})
        self.assertRaises(CouplingError, graph.distance, ('q0', 0), ('q1', 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
