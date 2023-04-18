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

# pylint: disable=missing-docstring

from qiskit.transpiler import MeasureGrouping
from qiskit.test import QiskitTestCase


class MeasureGroupingTest(QiskitTestCase):
    def test_empty_measuregrouping_class(self):
        meas_group = MeasureGrouping()
        self.assertEqual(None, meas_group.meas_map)

    def test_measuregrouping_class_with_meas_map_list(self):
        meas_group = MeasureGrouping(meas_map=[[0, 1], [1, 2], [3]])
        self.assertEqual({0: [0, 1], 1: [1, 2], 2: [1, 2], 3: [3]}, meas_group.meas_map)

    def test_measuregrouping_class_with_meas_map_dict(self):
        meas_map_dict = {0: [0, 1], 1: [1, 2], 2: [1, 2], 3: [3]}
        meas_group = MeasureGrouping(meas_map=meas_map_dict)
        self.assertEqual(meas_map_dict, meas_group.meas_map)

    def test_get_qubit_groups_with_empty_measuregrouping_class(self):
        meas_group = MeasureGrouping()
        qubits = [0, 1]
        self.assertEqual([0, 1], meas_group.get_qubit_groups(qubits))

    def test_get_qubit_groups_with_measuregrouping_class(self):
        meas_group = MeasureGrouping(meas_map=[[0, 1], [1, 2], [3]])
        qubits = [0, 1]
        self.assertEqual([0, 1, 2], meas_group.get_qubit_groups(qubits))
