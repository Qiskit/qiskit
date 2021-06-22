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
from qiskit.test.mock import FakeVigo
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import TrivialLayout


@ddt
class TestUnitarySynthesis(QiskitTestCase):
    """Test UnitarySynthesis pass."""

    def test_empty_basis_gates(self):
        """Verify when basis_gates is None, we do not synthesize unitaries."""
        qc = QuantumCircuit(1)
        qc.unitary([[0, 1], [1, 0]], [0])

        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(None).run(dag)

        self.assertEqual(out.count_ops(), {"unitary": 1})

    @data(
        ["u3", "cx"],
        ["u1", "u2", "u3", "cx"],
        ["rx", "ry", "rxx"],
        ["rx", "rz", "iswap"],
        ["u3", "rx", "rz", "cz", "iswap"],
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

    def test_two_qubit_synthesis_to_directional_cx_from_gate_errors(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap(conf.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_two_qubit_synthesis_to_directional_cx_multiple_registers(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr0 = QuantumRegister(1)
        qr1 = QuantumRegister(1)
        coupling_map = CouplingMap(conf.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr0, qr1)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_two_qubit_synthesis_to_directional_cx_from_coupling_map(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))


if __name__ == "__main__":
    unittest.main()
