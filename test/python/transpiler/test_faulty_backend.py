# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests preset pass manager whith faulty backends"""

from ddt import ddt, data

from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.converters import circuit_to_dag
from qiskit.test.mock import FakeOurenseFaultyQ1, FakeOurenseFaultyCX13
from qiskit.extensions.standard import CnotGate


class TestFaultyBackendCase(QiskitTestCase):
    def assertEqualCount(self, circuit1, circuit2):
        """Asserts circuit1 and circuit2 has the same result counts after execution in BasicAer"""
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 2048

        result1 = execute(circuit1, backend,
                          basis_gates=['u1', 'u2', 'u3', 'id', 'cx'],
                          seed_simulator=0, seed_transpiler=0, shots=shots).result().get_counts()

        result2 = execute(circuit2, backend,
                          basis_gates=['u1', 'u2', 'u3', 'id', 'cx'],
                          seed_simulator=0, seed_transpiler=0, shots=shots).result().get_counts()

        for key in set(result1.keys()).union(result2.keys()):
            with self.subTest(key=key):
                diff = abs(result1.get(key, 0) - result2.get(key, 0))
                self.assertLess(diff / shots * 100, 2.5)


@ddt
class TestFaultyCX01(TestFaultyBackendCase):
    """Test preset passmanagers with FakeOurenseFaultyCX01
    A fake 5 qubit backend, with a faulty CX(Q0, Q1) (and symmetric).
         0 (↔) 1 ↔ 3 ↔ 4
               ↕
               2
    """

    def setUp(self) -> None:
        self.backend = FakeOurenseFaultyCX13()

    def assertIdleCX01(self, circuit):
        """Asserts the CX(0, 1) (and symmetric) is not used in the circuit"""
        physical_qubits = QuantumRegister(5, 'q')
        cx_nodes = circuit_to_dag(circuit).op_nodes(CnotGate)
        for node in cx_nodes:
            if set(node.qargs) == {physical_qubits[0], physical_qubits[1]}:
                raise AssertionError('Faulty CX(Q0, Q1) (or symmetric) is being used.')

    @data(0, 1, 2, 3)
    def test_level(self, level):
        """Test level {level} Ourense backend with a faulty CX(Q0, Q1) """
        circuit = QuantumCircuit(QuantumRegister(4, 'qr'))
        circuit.h(range(4))
        circuit.ccx(0, 1, 2)
        circuit.measure_all()
        result = transpile(circuit,
                           backend=self.backend,
                           optimization_level=level,
                           seed_transpiler=42)

        self.assertIdleCX01(result)
        self.assertEqualCount(circuit, result)


@ddt
class TestFaultyQ1(TestFaultyBackendCase):
    """Test preset passmanagers with FakeOurenseFaultyQ1.
       A 5 qubit backend, with a faulty q1
         0 ↔ (1) ↔ 3 ↔ 4
              ↕
              2
    """

    def setUp(self) -> None:
        self.backend = FakeOurenseFaultyQ1()

    def assertIdleQ1(self, circuit):
        """Asserts the Q1 in circuit is not used with operations"""
        physical_qubits = QuantumRegister(5, 'q')
        nodes = circuit_to_dag(circuit).nodes_on_wire(physical_qubits[1])
        for node in nodes:
            if node.type == 'op':
                raise AssertionError('Faulty Qubit Q1 not totally idle')

    @data(0, 1, 2, 3)
    def test_level(self, level):
        """Test level {level} Ourense backend with a faulty Q1 """
        circuit = QuantumCircuit(QuantumRegister(2, 'qr'))
        circuit.h(range(2))
        circuit.cz(0, 1)
        circuit.measure_all()
        result = transpile(circuit, backend=self.backend,
                           optimization_level=level,
                           seed_transpiler=42)

        self.assertIdleQ1(result)
        self.assertEqualCount(circuit, result)
