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

# pylint: disable=missing-function-docstring

"""Tests the SetLayout pass"""

from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.test import QiskitTestCase


class TestSetLayout(QiskitTestCase):
    """Test the SetLayout pass."""

    def test_initial_layout_consistency_for_range_and_list(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc_1 = transpile(qc, coupling_map=cmap, initial_layout=range(3), seed_transpiler=42)
        tqc_2 = transpile(qc, coupling_map=cmap, initial_layout=list(range(3)), seed_transpiler=42)
        self.assertEqual(tqc_1.layout.initial_layout, tqc_2.layout.initial_layout)
