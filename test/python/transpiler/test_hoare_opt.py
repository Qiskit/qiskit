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

"""Test the HoareOptimizer pass"""

import unittest
from numpy import pi
from qiskit.transpiler.passes.optimization.hoare_opt import HAS_Z3
from qiskit.transpiler.passes import HoareOptimizer
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import XGate, RZGate, CSwapGate, SwapGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Statevector


@unittest.skipUnless(HAS_Z3, "z3-solver needs to be installed to run these tests")
class TestHoareOptimizer(QiskitTestCase):
    """Test the HoareOptimizer pass"""

    def test_phasegate_removal(self):
        """Should remove the phase on a classical state,
        but not on a superposition state.
        """
        circuit = QuantumCircuit(3)
        circuit.z(0)
        circuit.h(1)
        circuit.z(1)

        expected = QuantumCircuit(3)
        expected.h(1)
        expected.z(1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_cswap_removal(self):
        """Should remove Fredkin gates because the optimizer
        can deduce the targets are in the same state
        """
        circuit = QuantumCircuit(10)
        # prep
        circuit.x(0)
        circuit.h(3)
        circuit.h(4)
        circuit.h(5)
        circuit.h(6)
        # find first non-zero bit of reg(3-6), store position in reg(1-2)
        circuit.cx(3, 0)
        circuit.ccx(0, 4, 1)
        circuit.cx(1, 0)
        circuit.ccx(0, 5, 2)
        circuit.cx(2, 0)
        circuit.ccx(0, 6, 1)
        circuit.ccx(0, 6, 2)
        circuit.ccx(1, 2, 0)
        # shift circuit
        circuit.cswap(1, 7, 8)
        circuit.cswap(1, 8, 9)
        circuit.cswap(1, 9, 3)
        circuit.cswap(1, 3, 4)
        circuit.cswap(1, 4, 5)
        circuit.cswap(1, 5, 6)
        circuit.cswap(2, 7, 9)
        circuit.cswap(2, 8, 3)
        circuit.cswap(2, 9, 4)
        circuit.cswap(2, 3, 5)
        circuit.cswap(2, 4, 6)

        expected = QuantumCircuit(10)
        # prep
        expected.x(0)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        expected.h(6)
        # find first non-zero bit of reg(3-6), store position in reg(1-2)
        expected.cx(3, 0)
        expected.ccx(0, 4, 1)
        expected.cx(1, 0)
        expected.ccx(0, 5, 2)
        expected.cx(2, 0)
        expected.ccx(0, 6, 1)
        expected.ccx(0, 6, 2)
        expected.ccx(1, 2, 0)
        # optimized shift circuit
        expected.cswap(1, 3, 4)
        expected.cswap(1, 4, 5)
        expected.cswap(1, 5, 6)
        expected.cswap(2, 3, 5)
        expected.cswap(2, 4, 6)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_lnn_cnot_removal(self):
        """Should remove some cnots from swaps introduced
        because of linear nearest architecture. Only uses
        single-gate optimization techniques.
        """
        circuit = QuantumCircuit(5)
        circuit.h(0)
        for i in range(0, 3):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
            circuit.cx(i, i + 1)
        circuit.cx(3, 4)
        for i in range(3, 0, -1):
            circuit.cx(i - 1, i)
            circuit.cx(i, i - 1)

        expected = QuantumCircuit(5)
        expected.h(0)
        for i in range(0, 3):
            expected.cx(i, i + 1)
            expected.cx(i + 1, i)
        expected.cx(3, 4)
        for i in range(3, 0, -1):
            expected.cx(i, i - 1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_lnncnot_advanced_removal(self):
        """Should remove all cnots from swaps introduced
        because of linear nearest architecture. This time
        using multi-gate optimization techniques.
        """
        circuit = QuantumCircuit(5)
        circuit.h(0)
        for i in range(0, 3):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
            circuit.cx(i, i + 1)
        circuit.cx(3, 4)
        for i in range(3, 0, -1):
            circuit.cx(i - 1, i)
            circuit.cx(i, i - 1)

        expected = QuantumCircuit(5)
        expected.h(0)
        for i in range(0, 4):
            expected.cx(i, i + 1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=6)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_targetsuccessive_identity_removal(self):
        """Should remove pair of controlled target successive
        which are the inverse of each other, if they can be
        identified to be executed as a unit (either both or none).
        """
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.ccx(0, 1, 2)
        circuit.cx(1, 0)
        circuit.x(0)
        circuit.ccx(0, 1, 2)

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.cx(1, 0)
        expected.x(0)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=4)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_targetsuccessive_identity_advanced_removal(self):
        """Should remove target successive identity gates
        with DIFFERENT sets of control qubits.
        In this case CCCX(4,5,6,7) & CCX(5,6,7).
        """
        circuit = QuantumCircuit(8)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.h(3)
        circuit.h(4)
        circuit.h(5)
        for i in range(3):
            circuit.cx(i * 2 + 1, i * 2)
        circuit.cx(3, 5)
        for i in range(2):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
            circuit.cx(i * 2 + 3, i * 2 + 2)
        circuit.ccx(4, 5, 6)
        for i in range(1, -1, -1):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(3, 5)
        circuit.cx(5, 6)
        circuit.cx(3, 5)
        circuit.x(6)
        for i in range(2):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        for i in range(1, -1, -1):
            circuit.cx(i * 2 + 3, i * 2 + 2)
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(1, 0)
        circuit.ccx(6, 1, 0)
        circuit.ccx(0, 1, 3)
        circuit.ccx(6, 3, 2)
        circuit.ccx(2, 3, 5)
        circuit.ccx(6, 5, 4)
        circuit.append(XGate().control(3), [4, 5, 6, 7], [])
        for i in range(1, -1, -1):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(3, 5)
        for i in range(1, 3):
            circuit.cx(i * 2 + 1, i * 2)
        circuit.ccx(5, 6, 7)

        expected = QuantumCircuit(8)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        for i in range(3):
            expected.cx(i * 2 + 1, i * 2)
        expected.cx(3, 5)
        for i in range(2):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
            expected.cx(i * 2 + 3, i * 2 + 2)
        expected.ccx(4, 5, 6)
        for i in range(1, -1, -1):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(3, 5)
        expected.cx(5, 6)
        expected.cx(3, 5)
        expected.x(6)
        for i in range(2):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        for i in range(1, -1, -1):
            expected.cx(i * 2 + 3, i * 2 + 2)
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(1, 0)
        expected.ccx(6, 1, 0)
        expected.ccx(0, 1, 3)
        expected.ccx(6, 3, 2)
        expected.ccx(2, 3, 5)
        expected.ccx(6, 5, 4)
        for i in range(1, -1, -1):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(3, 5)
        for i in range(1, 3):
            expected.cx(i * 2 + 1, i * 2)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_control_removal(self):
        """Should replace CX by X and CZ by Z."""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.cx(0, 1)

        expected = QuantumCircuit(2)
        expected.x(0)
        expected.x(1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv @ circuit, stv @ expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.x(1)
        circuit.cz(0, 1)
        circuit.h(0)

        expected = QuantumCircuit(2)
        expected.h(0)
        expected.x(1)
        expected.z(0)
        expected.h(0)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv @ circuit, stv @ expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_is_identity(self):
        """The is_identity function determines whether a pair of gates
        forms the identity, when ignoring control qubits.
        """
        seq = [DAGOpNode(op=XGate().control()), DAGOpNode(op=XGate().control(2))]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

        seq = [
            DAGOpNode(op=RZGate(-pi / 2).control()),
            DAGOpNode(op=RZGate(pi / 2).control(2)),
        ]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

        seq = [DAGOpNode(op=CSwapGate()), DAGOpNode(op=SwapGate())]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

    def test_multiple_pass(self):
        """Verify that multiple pass can be run
        with the same Hoare instance.
        """
        circuit1 = QuantumCircuit(2)
        circuit1.z(0)
        circuit1.h(1)
        circuit1.z(1)

        circuit2 = QuantumCircuit(2)
        circuit2.z(1)
        circuit2.h(0)
        circuit2.z(0)

        expected = QuantumCircuit(2)
        expected.h(0)
        expected.z(0)

        pass_ = HoareOptimizer()

        pass_.run(circuit_to_dag(circuit1))
        result2 = pass_.run(circuit_to_dag(circuit2))

        self.assertEqual(result2, circuit_to_dag(expected))


if __name__ == "__main__":
    unittest.main()
