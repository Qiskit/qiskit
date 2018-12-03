# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for transpiler functionality"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import compile, Aer
from qiskit.transpiler import PassManager, transpile_dag, transpile
from qiskit.tools._compiler import circuits_to_qobj
from qiskit.transpiler.passes import CXCancellation
from qiskit.dagcircuit import DAGCircuit
from .common import QiskitTestCase


class TestTranspiler(QiskitTestCase):
    """Test combining and extending of QuantumCircuits."""

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
        dag_circuit = DAGCircuit.fromQuantumCircuit(circuit)
        resources_before = dag_circuit.count_ops()

        pass_manager = PassManager()
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, direction_mapper, cx cancellation, optimize_1q_gates
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
        basis_gates = 'u1,u2,u3,cx,id'

        backend = Aer.get_backend('qasm_simulator_py')
        circuit2 = transpile(circuit, backend, coupling_map=coupling_map, basis_gates=basis_gates,
                             pass_manager=None)

        qobj = compile(circuit, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates)

        qobj2 = circuits_to_qobj(circuit2, backend.name(), basis_gates=basis_gates,
                                 coupling_map=coupling_map, qobj_id=qobj.qobj_id)

        self.assertEqual(qobj, qobj2)

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation pass.

        It should cancel consecutive cx pairs on same qubits.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag_circuit = DAGCircuit.fromQuantumCircuit(circuit)

        pass_manager = PassManager()
        pass_manager.add_passes(CXCancellation())
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertNotIn('cx', resources_after)
