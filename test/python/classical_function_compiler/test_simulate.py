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
from ddt import ddt, data
from qiskit.circuit.classicalfunction import classical_function as compile_classical_function
from qiskit.test import QiskitTestCase
from .utils import get_truthtable_from_function, example_list


@ddt
class TestSimulate(QiskitTestCase):
    """Tests LogicNetwork.simulate method"""

    @data(*example_list())
    def test_(self, a_callable):
        """Tests LogicSimulate.simulate() on all the examples"""
        network = compile_classical_function(a_callable)
        truth_table = network.simulate_all()
        self.assertEqual(truth_table, get_truthtable_from_function(a_callable))
