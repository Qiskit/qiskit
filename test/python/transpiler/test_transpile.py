# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests basic functionality of the transpile function"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import compile, BasicAer
from qiskit.transpiler import PassManager, transpile_dag, transpile
from qiskit.compiler import assemble_circuits
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
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
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        coupling_map = [[1, 0]]
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

        backend = BasicAer.get_backend('qasm_simulator')
        circuit2 = transpile(circuit, backend, coupling_map=coupling_map, basis_gates=basis_gates,
                             pass_manager=None)

        qobj = compile(circuit, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates)
        run_config = RunConfig(shots=1024, max_credits=10)
        qobj2 = assemble_circuits(circuit2, qobj_id=qobj.qobj_id, run_config=run_config)
        self.assertEqual(qobj, qobj2)
