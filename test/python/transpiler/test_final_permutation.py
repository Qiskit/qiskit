# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""FinalPermutation in transpile flow pass testing"""

import unittest

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, StarPreRouting
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from utils import random_circuit


class TestFinalPermutationInTranspile(QiskitTestCase):
    """Tests for FixedPoint pass."""

    def test_sabre_1(self):
        """SabreLayout"""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)

        op = Operator(qc)

        cm = CouplingMap.from_line(6)
        cm.make_symmetric()
        print(cm)

        qct = transpile(qc, optimization_level=3, coupling_map=cm, basis_gates=["cx", "u"])
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_sabre_2(self):
        """Sabre."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        op = Operator(qc)

        cm = CouplingMap.from_line(6)

        qct = transpile(qc, optimization_level=3, coupling_map=cm, basis_gates=["cx", "u"])
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_sabre_3(self):
        """Sabre."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        op = Operator(qc)

        cm = CouplingMap.from_line(6)

        qct = transpile(qc, optimization_level=3, coupling_map=cm, basis_gates=["cx", "u"])
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_sabre_1(self):
        """Circuit with PermutationGate. ElidePermutations."""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        op = Operator(qc)

        cm = CouplingMap.from_line(6)
        cm.make_symmetric()

        qct = transpile(qc, optimization_level=3, coupling_map=cm, basis_gates=["cx", "u"])
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_sabre_2(self):
        """This runs ElidePermutations + Sabre"""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.swap(0, 1)
        qc.cx(1, 5)
        qc.cx(3, 5)
        op = Operator(qc)

        cm = CouplingMap.from_line(6)
        cm.make_symmetric()
        print(cm)

        qct = transpile(qc, optimization_level=3, coupling_map=cm, basis_gates=["cx", "u"])
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_vf2(self):
        "StarPreRouting + perfect map"
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        op = Operator(qc)

        cm = CouplingMap.from_line(6)

        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=cm,
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        qct = spm.run(qc)
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_star_sabre(self):
        """Both StarPreRouting+Elide."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        op = Operator(qc)
        coupling_map = CouplingMap.from_line(5)

        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=coupling_map,
            basis_gates=["u", "cz"],
        )
        spm.init += ElidePermutations()
        spm.init += StarPreRouting()
        qct = spm.run(qc)
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_elide_sabre(self):
        """Both StarPreRouting+Elide."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        op = Operator(qc)
        coupling_map = CouplingMap.from_line(5)

        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=coupling_map,
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        spm.init += ElidePermutations()
        qct = spm.run(qc)
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_elide_sabre_2(self):
        """First StarPreRouting, then Elide."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        op = Operator(qc)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=CouplingMap.from_line(6),
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        spm.init += ElidePermutations()
        qct = spm.run(qc)
        # self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_elide_3(self):
        """First StarPreRouting, then Elide."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        op = Operator(qc)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=CouplingMap.from_line(5),
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        spm.init += ElidePermutations()
        qct = spm.run(qc)
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_random_embed_1(self):
        """More qubits in coupling map"""
        for seed in range(10):
            qc = random_circuit(4, 5, 3, seed=seed)
            nq = qc.num_qubits

            coupling_map = CouplingMap.from_line(nq + 2)

            qc2 = QuantumCircuit(nq + 2)
            qc2 = qc2.compose(qc, range(nq))

            extended_op = Operator(qc2)

            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=coupling_map,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
            )

            transpiled_op = Operator.from_circuit(qct)
            transpiled_op_new = Operator._from_circuit_new(qct)

            self.assertTrue(transpiled_op.equiv(extended_op))
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_routing_methods(self):
        """Stochastic Swap for mapping (sets final layout)"""
        qc = random_circuit(4, 5, 3, seed=1)
        nq = qc.num_qubits
        op = Operator(qc)
        cm = CouplingMap.from_line(nq + 2)
        qc2 = QuantumCircuit(nq + 2)
        qc2 = qc2.compose(qc, range(nq))
        extended_op = Operator(qc2)

        for routing_method in ["stochastic", "lookahead", "basic", "sabre"]:
            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=cm,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
                routing_method=routing_method,
            )

            transpiled_op = Operator.from_circuit(qct)
            transpiled_op_new = Operator._from_circuit_new(qct)

            self.assertTrue(transpiled_op.equiv(extended_op))
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_elide_and_routing_methods(self):
        """Stochastic Swap for mapping (sets final layout)"""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        nq = qc.num_qubits
        op = Operator(qc)
        cm = CouplingMap.from_line(nq + 2)
        qc2 = QuantumCircuit(nq + 2)
        qc2 = qc2.compose(qc, range(nq))
        extended_op = Operator(qc2)

        for routing_method in ["stochastic", "lookahead", "basic", "sabre"]:
            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=cm,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
                routing_method=routing_method,
            )

            transpiled_op = Operator.from_circuit(qct)
            transpiled_op_new = Operator._from_circuit_new(qct)

            self.assertTrue(transpiled_op.equiv(extended_op))
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_with_post_layout(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        op = Operator(qc)

        qc2 = QuantumCircuit(5)
        qc2 = qc2.compose(qc, range(3))
        extended_op = Operator(qc2)

        coupling_map = CouplingMap.from_line(5)
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "sx", "x", "cx", "rz"],
            coupling_map=coupling_map,
            seed=0,
        )
        qct = transpile(qc, backend=backend, seed_transpiler=4242)

        transpiled_op = Operator.from_circuit(qct)
        transpiled_op_new = Operator._from_circuit_new(qct)

        self.assertTrue(transpiled_op.equiv(extended_op))
        self.assertTrue(transpiled_op_new.equiv(extended_op))


if __name__ == "__main__":
    unittest.main()
