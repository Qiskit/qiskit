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

"""
Tests for the UnitarySynthesis transpiler pass.
"""

import unittest

from ddt import ddt, data

from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.quantum_info.operators import Operator


@ddt
class TestUnitarySynthesis(QiskitTestCase):
    """Test UnitarySynthesis pass."""

    def test_empty_basis_gates(self):
        """Verify when basis_gates is None, we do not synthesize unitaries."""
        qc = QuantumCircuit(1)
        qc.unitary([[0, 1], [1, 0]], [0])

        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(None).run(dag)

        self.assertEqual(out.count_ops(), {'unitary': 1})

    @data(
        ['u3', 'cx'],
        ['u1', 'u2', 'u3', 'cx'],
        ['rx', 'ry', 'rxx'],
        ['rx', 'rz', 'iswap'],
        ['u3', 'rx', 'rz', 'cz', 'iswap'],
    )
    def test_two_qubit_synthesis_to_basis(self, basis_gates):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell_op = Operator(bell)

        qc = QuantumCircuit(2)
        qc.unitary(bell_op, [0, 1])
        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(basis_gates).run(dag)

        self.assertTrue(set(out.count_ops()).issubset(basis_gates))


if __name__ == '__main__':
    unittest.main()
