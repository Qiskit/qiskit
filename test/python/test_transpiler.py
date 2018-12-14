# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for transpiler functionality"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import compile, BasicAer
from qiskit.transpiler import PassManager, transpile_dag, transpile
from qiskit.tools.compiler import circuits_to_qobj
from qiskit.transpiler.passes import CXCancellation, LookaheadMapper
from qiskit.converters import circuit_to_dag
from qiskit.mapper import Coupling

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
        dag_circuit = circuit_to_dag(circuit)
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

        backend = BasicAer.get_backend('qasm_simulator')
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
        dag_circuit = circuit_to_dag(circuit)

        pass_manager = PassManager()
        pass_manager.add_passes(CXCancellation())
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertNotIn('cx', resources_after)

    def test_lookahead_mapper_doesnt_modify_mapped_circuit(self):
        """Test that lookahead mapper is idempotent.

        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        original_dag = circuit_to_dag(circuit)

        # Create coupling map which contains all two-qubit gates in the circuit.
        coupling_map = Coupling(couplinglist=[(0, 1), (0, 2)])

        pass_manager = PassManager()
        pass_manager.add_passes(LookaheadMapper(coupling_map))
        mapped_dag = transpile_dag(original_dag, pass_manager=pass_manager)

        self.assertEqual(original_dag, mapped_dag)

        second_pass_manager = PassManager()
        second_pass_manager.add_passes(LookaheadMapper(coupling_map))
        remapped_dag = transpile_dag(mapped_dag, pass_manager=second_pass_manager)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_lookahead_mapper_should_add_a_single_swap(self):
        """Test that lookahead mapper will insert a SWAP to match layout.

        For a single cx gate which is not available in the current layout, test
        that the mapper inserts a single swap to enable the gate.
        """

        qr = QuantumRegister(3)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)

        coupling_map = Coupling(couplinglist=[(0, 1), (1, 2)])

        pass_manager = PassManager()
        pass_manager.add_passes([LookaheadMapper(coupling_map)])
        mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)

        self.assertEqual(mapped_dag.count_ops().get('swap', 0),
                         dag_circuit.count_ops().get('swap', 0) + 1)

    def test_lookahead_mapper_finds_minimal_swap_solution(self):
        """Of many valid SWAPs, test that lookahead finds the cheapest path.

        For a two CNOT circuit: cx q[0],q[2]; cx q[0],q[1]
        on the initial layout: qN -> qN
        (At least) two solutions exist:
        - SWAP q[0],[1], cx q[0],q[2], cx q[0],q[1]
        - SWAP q[1],[2], cx q[0],q[2], SWAP q[1],q[2], cx q[0],q[1]

        Verify that we find the first solution, as it requires fewer SWAPs.
        """

        qr = QuantumRegister(3)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = Coupling(couplinglist=[(0, 1), (1, 2)])

        pass_manager = PassManager()
        pass_manager.add_passes([LookaheadMapper(coupling_map)])
        mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)

        self.assertEqual(mapped_dag.count_ops().get('swap', 0),
                         dag_circuit.count_ops().get('swap', 0) + 1)
