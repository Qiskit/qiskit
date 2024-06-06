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
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, StarPreRouting
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestFinalPermutationInTranspile(QiskitTestCase):
    """Tests for FixedPoint pass."""

    def test1(self):
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
        self.assertTrue(Operator.from_circuit_new(qct).equiv(op))

    def test2(self):
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
        self.assertTrue(Operator.from_circuit_new(qct).equiv(op))

    def test3(self):
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
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=CouplingMap.from_line(5),
            basis_gates=["u", "cz"],
        )
        spm.init += ElidePermutations()
        spm.init += StarPreRouting()
        qct = spm.run(qc)
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator.from_circuit_new(qct).equiv(op))


if __name__ == "__main__":
    unittest.main()

