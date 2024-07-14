# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests LogicNetwork.simulate method."""

import unittest
from ddt import ddt, data
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from test import QiskitTestCase  # pylint: disable=wrong-import-order

from . import utils

if HAS_TWEEDLEDUM:
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
@ddt
class TestSimulate(QiskitTestCase):
    # pylint: disable=possibly-used-before-assignment
    """Tests LogicNetwork.simulate method"""

    @data(*utils.example_list())
    def test_(self, a_callable):
        """Tests LogicSimulate.simulate() on all the examples"""
        network = compile_classical_function(a_callable)
        truth_table = network.simulate_all()
        self.assertEqual(truth_table, utils.get_truthtable_from_function(a_callable))
