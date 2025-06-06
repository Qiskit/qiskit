# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test basic simulator."""

import os
import unittest
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


from . import BasicProviderBackendTestMixin


class TestBasicSimulator(QiskitTestCase, BasicProviderBackendTestMixin):
    """Test the basic provider simulator."""

    def setUp(self):
        super().setUp()
        self.backend = BasicSimulator()
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])
        self.circuit = bell

        self.seed = 88
        qasm_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "qasm"
        )
        qasm_filename = os.path.join(qasm_dir, "example.qasm")
        qcirc = QuantumCircuit.from_qasm_file(qasm_filename)
        qcirc.name = "test"
        self.transpiled_circuit = transpile(qcirc, backend=self.backend)

    def test_basic_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        result = self.backend.run(
            self.transpiled_circuit, shots=shots, seed_simulator=self.seed
        ).result()
        self.assertEqual(result.success, True)

    def test_measure_sampler_repeated_qubits(self):
        """Test measure sampler if qubits measured more than once."""
        shots = 100
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(4, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[1], cr[2])
        circuit.measure(qr[0], cr[3])
        target = {"0110": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_measure_sampler_single_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(1, "cr")

        for qubit in range(num_qubits):
            circuit = QuantumCircuit(qr, cr)
            circuit.x(qr[qubit])
            circuit.measure(qr[qubit], cr[0])
            target = {"1": shots}
            job = self.backend.run(
                transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
            )
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target)

    def test_measure_sampler_partial_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(4, "cr")

        #             ░     ░     ░ ┌─┐ ░
        # qr_0: ──────░─────░─────░─┤M├─░────
        #       ┌───┐ ░     ░ ┌─┐ ░ └╥┘ ░
        # qr_1: ┤ X ├─░─────░─┤M├─░──╫──░────
        #       └───┘ ░     ░ └╥┘ ░  ║  ░
        # qr_2: ──────░─────░──╫──░──╫──░────
        #       ┌───┐ ░ ┌─┐ ░  ║  ░  ║  ░ ┌─┐
        # qr_3: ┤ X ├─░─┤M├─░──╫──░──╫──░─┤M├
        #       └───┘ ░ └╥┘ ░  ║  ░  ║  ░ └╥┘
        # qr_4: ──────░──╫──░──╫──░──╫──░──╫─
        #             ░  ║  ░  ║  ░  ║  ░  ║
        # cr: 4/═════════╩═════╩═════╩═════╩═
        #                1     0     2     3
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[3])
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[1])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[2])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[3])
        target = {"1011": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_basic_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = self.backend.run(
            self.transpiled_circuit, shots=1000, seed_simulator=self.seed
        ).result()
        shots = 1024
        threshold = 0.04 * shots
        counts = result.get_counts("test")
        target = {
            "100 100": shots / 8,
            "011 011": shots / 8,
            "101 101": shots / 8,
            "111 111": shots / 8,
            "000 000": shots / 8,
            "010 010": shots / 8,
            "110 110": shots / 8,
            "001 001": shots / 8,
        }
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_memory(self):
        """Test memory."""
        #       ┌───┐        ┌─┐
        # qr_0: ┤ H ├──■─────┤M├───
        #       └───┘┌─┴─┐   └╥┘┌─┐
        # qr_1: ─────┤ X ├────╫─┤M├
        #            └┬─┬┘    ║ └╥┘
        # qr_2: ──────┤M├─────╫──╫─
        #       ┌───┐ └╥┘ ┌─┐ ║  ║
        # qr_3: ┤ X ├──╫──┤M├─╫──╫─
        #       └───┘  ║  └╥┘ ║  ║
        # cr0: 2/══════╬═══╬══╩══╩═
        #              ║   ║  0  1
        #              ║   ║
        # cr1: 2/══════╩═══╩═══════
        #              0   1
        qr = QuantumRegister(4, "qr")
        cr0 = ClassicalRegister(2, "cr0")
        cr1 = ClassicalRegister(2, "cr1")
        circ = QuantumCircuit(qr, cr0, cr1)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[3])
        circ.measure(qr[0], cr0[0])
        circ.measure(qr[1], cr0[1])
        circ.measure(qr[2], cr1[0])
        circ.measure(qr[3], cr1[1])

        shots = 50
        job = self.backend.run(
            transpile(circ, self.backend), shots=shots, seed_simulator=self.seed, memory=True
        )
        result = job.result()
        memory = result.get_memory()
        self.assertEqual(len(memory), shots)
        for mem in memory:
            self.assertIn(mem, ["10 00", "10 11"])

    def test_unitary(self):
        """Test unitary gate instruction"""
        max_qubits = 4
        x_mat = np.array([[0, 1], [1, 0]])
        # Test 1 to max_qubits for random n-qubit unitary gate
        for i in range(max_qubits):
            num_qubits = i + 1
            # Apply X gate to all qubits
            multi_x = x_mat
            for _ in range(i):
                multi_x = np.kron(multi_x, x_mat)
            # Target counts
            shots = 1024
            target_counts = {num_qubits * "1": shots}
            # Test circuit
            qr = QuantumRegister(num_qubits, "qr")
            cr = ClassicalRegister(num_qubits, "cr")
            circuit = QuantumCircuit(qr, cr)
            circuit.unitary(multi_x, qr)
            circuit.measure(qr, cr)
            job = self.backend.run(transpile(circuit, self.backend), shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target_counts)

    def test_options(self):
        """Test setting custom backend options during init and run."""
        init_statevector = np.zeros(2**2, dtype=complex)
        init_statevector[2] = 1
        in_options = {
            "initial_statevector": init_statevector,
            "seed_simulator": 42,
            "shots": 100,
            "memory": True,
        }
        backend = BasicSimulator()
        backend_with_options = BasicSimulator(
            initial_statevector=in_options["initial_statevector"],
            seed_simulator=in_options["seed_simulator"],
            shots=in_options["shots"],
            memory=in_options["memory"],
        )
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])

        with self.subTest(msg="Test init options"):
            out_options = backend_with_options.options
            for key in out_options:
                if key != "initial_statevector":
                    self.assertEqual(getattr(out_options, key), in_options.get(key))
                else:
                    np.testing.assert_array_equal(getattr(out_options, key), in_options.get(key))

        with self.subTest(msg="Test run options"):
            out_1 = backend_with_options.run(bell).result().get_counts()
            out_2 = (
                backend.run(
                    bell,
                    initial_statevector=in_options["initial_statevector"],
                    seed_simulator=in_options["seed_simulator"],
                    shots=in_options["shots"],
                    memory=in_options["memory"],
                )
                .result()
                .get_counts()
            )
            self.assertEqual(out_1, out_2)

        with self.subTest(msg="Test run options don't overwrite init"):
            init_statevector = np.zeros(2**2, dtype=complex)
            init_statevector[3] = 1
            other_options = {
                "initial_statevector": init_statevector,
                "seed_simulator": 0,
                "shots": 1000,
                "memory": True,
            }
            out_1 = backend_with_options.run(bell).result().get_counts()
            out_2 = (
                backend_with_options.run(
                    bell,
                    initial_statevector=other_options["initial_statevector"],
                    seed_simulator=other_options["seed_simulator"],
                    shots=other_options["shots"],
                    memory=other_options["memory"],
                )
                .result()
                .get_counts()
            )
            self.assertNotEqual(out_1, out_2)
            out_options = backend_with_options.options
            for key in out_options:
                if key != "initial_statevector":
                    self.assertEqual(getattr(out_options, key), in_options.get(key))
                else:
                    np.testing.assert_array_equal(getattr(out_options, key), in_options.get(key))


if __name__ == "__main__":
    unittest.main()
