# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests basic functionality of the transpile function"""

import math

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import compile, BasicAer
from qiskit.extensions.standard import CnotGate
from qiskit.transpiler import PassManager, transpile_dag, transpile
from qiskit.compiler import assemble_circuits
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.compiler import RunConfig


class TestTranspile(QiskitTestCase):
    """Test transpile function."""

    def test_pass_manager_empty(self):
        """Test passing an empty PassManager() to the transpiler.

        It should perform no transformations on the circuit.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        dag_circuit = circuit_to_dag(circuit)
        resources_before = dag_circuit.count_ops()

        pass_manager = PassManager()
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, cx_direction, cx_cancellation, optimize_1q_gates
        and should be equivalent to using tools.compile
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        coupling_map = [[1, 0]]
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

        backend = BasicAer.get_backend('qasm_simulator')
        circuit2 = transpile(circuit, backend, coupling_map=coupling_map, basis_gates=basis_gates,
                             pass_manager=None)

        qobj = compile(circuit, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates)
        run_config = RunConfig(shots=1024, max_credits=10)
        qobj2 = assemble_circuits(circuit2, qobj_id=qobj.qobj_id, run_config=run_config)
        self.assertEqual(qobj, qobj2)

    def test_transpile_basis_gates_no_backend_no_coupling_map(self):
        """Verify tranpile() works with no coupling_map or backend."""
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']
        circuit2 = transpile(circuit, basis_gates=basis_gates)
        dag_circuit = circuit_to_dag(circuit2)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({'u2': 2, 'cx': 4}, resources_after)

    def test_transpile_non_adjacent_layout(self):
        """Transpile pipeline can handle manual layout on non-adjacent qubits.

        circuit:
        qr0:-[H]--.------------  -> 1
                  |
        qr1:-----(+)--.--------  -> 2
                      |
        qr2:---------(+)--.----  -> 3
                          |
        qr3:-------------(+)---  -> 5

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        new_circuit = transpile(circuit,
                                basis_gates=basis_gates,
                                coupling_map=coupling_map,
                                initial_layout=initial_layout)

        for gate, qargs, _ in new_circuit.data:
            if isinstance(gate, CnotGate):
                self.assertIn([x[1] for x in qargs], coupling_map)

    def test_transpile_qft_grid(self):
        """Transpile pipeline can handle 8-qubit QFT on 14-qubit grid.
        """
        qr = QuantumRegister(8)
        circuit = QuantumCircuit(qr)
        for i, _ in enumerate(qr):
            for j in range(i):
                circuit.cu1(math.pi/float(2**(i-j)), qr[i], qr[j])
            circuit.h(qr[i])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        new_circuit = transpile(circuit,
                                basis_gates=basis_gates,
                                coupling_map=coupling_map)

        for gate, qargs, _ in new_circuit.data:
            if isinstance(gate, CnotGate):
                self.assertIn([x[1] for x in qargs], coupling_map)
