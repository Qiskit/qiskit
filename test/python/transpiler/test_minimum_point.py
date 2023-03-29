# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MinimumPoint pass testing"""

from qiskit.transpiler.passes import MinimumPoint
from qiskit.dagcircuit import DAGCircuit
from qiskit.test import QiskitTestCase


class TestMinimumPointtPass(QiskitTestCase):
    """Tests for MinimumPoint pass."""

    def test_minimum_point_reached_fixed_point_single_field(self):
        """Test a fixed point is reached with a single field."""
        min_pass = MinimumPoint(["depth"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["depth"] = 42
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertIsNone(min_pass.property_set["test_backtrack_history"])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((42,), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((42,), min_pass.property_set["test_backtrack_history"][0])
        self.assertTrue(min_pass.property_set["test_minimum_point"])

    def test_minimum_point_reached_fixed_point_multiple_fields(self):
        """Test a fixed point is reached with a multiple fields."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertIsNone(min_pass.property_set["test_backtrack_history"])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((0.875, 15, 20), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((0.875, 15, 20), min_pass.property_set["test_backtrack_history"][0])
        self.assertTrue(min_pass.property_set["test_minimum_point"])

    def test_min_over_backtrack_range(self):
        """Test minimum returned over backtrack depth."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertIsNone(min_pass.property_set["test_backtrack_history"])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 2)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 3)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 4)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 5)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertTrue(min_pass.property_set["test_minimum_point"])

    def test_min_reset_backtrack_range(self):
        """Test minimum resets backtrack depth."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertIsNone(min_pass.property_set["test_backtrack_history"])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 2)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 3)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 4)
        self.assertEqual((0.775, 25, 35), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 10
        min_pass.property_set["size"] = 10
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 1)
        self.assertEqual((0.775, 10, 10), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 2)
        self.assertEqual((0.775, 10, 10), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_min_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 3)
        self.assertEqual((0.775, 10, 10), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 4)
        self.assertEqual((0.775, 10, 10), min_pass.property_set["test_backtrack_history"][0])
        self.assertIsNone(min_pass.property_set["test_minimum_point"])
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        self.assertEqual(min_pass.property_set["test_minimum_point_count"], 5)
        self.assertEqual((0.775, 10, 10), min_pass.property_set["test_backtrack_history"][0])
        self.assertTrue(min_pass.property_set["test_minimum_point"])
