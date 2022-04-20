# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler passes that deal with Cliffords."""

import unittest

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.transpiler.passes.optimization.optimize_cliffords import OptimizeCliffords
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators import Clifford
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator


class TestCliffordPasses(QiskitTestCase):
    """Tests to verify correctness of the transpiler passes that deal with Cliffords."""

    def create_cliff1(self):
        """Creates a simple Clifford."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff2(self):
        """Creates another simple Clifford."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff3(self):
        """Creates a third Clifford which is the composition of the previous two."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def test_circuit_with_cliffords(self):
        """Test that Cliffords get stored natively on a QuantumCircuit,
        and that QuantumCircuit's decompose() replaces Clifford with gates."""

        # Create a circuit with 2 cliffords and four other gates
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(2, 0)
        qc.append(cliff1, [3, 0, 2])
        qc.swap(1, 3)
        qc.append(cliff2, [1, 2, 3])
        qc.h(3)
        # print(qc)

        # Check that there are indeed two Clifford objects in the circuit,
        # and that these are not gates.
        cliffords = [inst for inst, _, _ in qc.data if isinstance(inst, Clifford)]
        gates = [inst for inst, _, _ in qc.data if isinstance(inst, Gate)]
        self.assertEqual(len(cliffords), 2)
        self.assertEqual(len(gates), 4)

        # Check that calling QuantumCircuit's decompose(), no Clifford objects remain
        qc2 = qc.decompose()
        # print(qc2)
        cliffords2 = [inst for inst, _, _ in qc2.data if isinstance(inst, Clifford)]
        self.assertEqual(len(cliffords2), 0)

    def test_can_construct_operator(self):
        """Test that we can construct an Operator from a circuit that
        contains a Clifford gate."""

        cliff = self.create_cliff1()
        qc = QuantumCircuit(4)
        qc.append(cliff, [3, 1, 2])

        # Create an operator from the decomposition of qc into gates
        op1 = Operator(qc.decompose())

        # Create an operator from qc directly
        op2 = Operator(qc)

        # Check that the two operators are equal
        self.assertTrue(op1.equiv(op2))

    def test_can_combine_cliffords(self):
        """Test that we can combine a pair of Cliffords over the same qubits
        using OptimizeCliffords transpiler pass."""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        cliff3 = self.create_cliff3()

        # Create a circuit with two consective cliffords
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 1, 2])
        self.assertEqual(qc1.count_ops()["clifford"], 2)

        # Run OptimizeCliffords pass, and check that only one Clifford remains
        qc1opt = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1opt.count_ops()["clifford"], 1)

        # Create the expected circuit
        qc2 = QuantumCircuit(4)
        qc2.append(cliff3, [3, 1, 2])

        # Check that all possible operators are equal
        self.assertTrue(Operator(qc1).equiv(Operator(qc1.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc1opt.decompose())))
        self.assertTrue(Operator(qc1).equiv(Operator(qc1opt)))
        self.assertTrue(Operator(qc2).equiv(Operator(qc2.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc2)))

    def test_cannot_combine(self):
        """Test that currently we cannot combine a pair of Cliffords.
        The result will be changed after pass is updated"""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 2, 1])
        qc1 = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1.count_ops()["clifford"], 2)


if __name__ == "__main__":
    unittest.main()
