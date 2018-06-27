# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Tests for transpiler functionality"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import wrapper
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import CXCancellation
from qiskit.dagcircuit import DAGCircuit
from qiskit.unroll import DagUnroller, JsonBackend
from .common import QiskitTestCase


class TestTranspiler(QiskitTestCase):
    """Test combining and extending of QuantumCircuits."""

    def test_pass_manager_empty(self):
        """Test passing an empty PassManager() to the transpiler.

        It should perform no transformations on the circuit.
        """
        q = QuantumRegister(2)
        circ = QuantumCircuit(q)
        circ.h(q[0])
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)
        resources_before = dag_circuit.count_ops()

        pass_manager = PassManager()
        dag_circuit = transpile(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, direction_mapper, cx cancellation, optimize_1q_gates
        and should be equivalent to using wrapper.compile
        """
        q = QuantumRegister(2)
        circ = QuantumCircuit(q)
        circ.h(q[0])
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])

        coupling_map = [[1, 0]]
        basis_gates = 'u1,u2,u3,cx,id'

        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)
        dag_circuit = transpile(dag_circuit, coupling_map=coupling_map,
                                basis_gates=basis_gates, pass_manager=None)
        transpiler_json = DagUnroller(dag_circuit, JsonBackend(dag_circuit.basis)).execute()

        qobj = wrapper.compile(circ, backend='local_qasm_simulator',
                               coupling_map=coupling_map, basis_gates=basis_gates)
        compiler_json = qobj['circuits'][0]['compiled_circuit']

        self.assertDictEqual(transpiler_json, compiler_json)

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation pass.

        It should cancel consecutive cx pairs on same qubits.
        """
        q = QuantumRegister(2)
        circ = QuantumCircuit(q)
        circ.h(q[0])
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[0], q[1])
        circ.cx(q[1], q[0])
        circ.cx(q[1], q[0])
        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)

        pass_manager = PassManager()
        pass_manager.add_pass(CXCancellation())
        dag_circuit = transpile(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertNotIn('cx', resources_after)
