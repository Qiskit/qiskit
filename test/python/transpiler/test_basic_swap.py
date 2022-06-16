# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the BasicSwap pass"""

import unittest
from ddt import ddt, data
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.analysis.distance import hellinger_distance
from qiskit import Aer


@ddt
class TestBasicSwap(QiskitTestCase):
    """Tests the BasicSwap pass."""

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
        q0:--(+)-[U]-(+)-
              |       |
        q1:---.-------|--
                      |
        q2:-----------.--

        CouplingMap map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """No need to have any swap, two CXs distance 1 to each other, in the same layer
        q0:--(+)--
              |
        q1:---.---

        q2:--(+)--
              |
        q3:---.---

        CouplingMap map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        """Adding a swap
        q0:-------

        q1:--(+)--
              |
        q2:---.---

        CouplingMap map: [1]--[0]--[2]

        q0:--X---.---
             |   |
        q1:--X---|---
                 |
        q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_a_single_swap_bigger_cm(self):
        """Swapper in a bigger coupling map
        q0:-------

        q1:---.---
              |
        q2:--(+)--

        CouplingMap map: [1]--[0]--[2]--[3]

        q0:--X---.---
             |   |
        q1:--X---|---
                 |
        q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [0, 2], [2, 3]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_keep_layout(self):
        """After a swap, the following gates also change the wires.
        qr0:---.---[H]--
               |
        qr1:---|--------
               |
        qr2:--(+)-------

        CouplingMap map: [0]--[1]--[2]

        qr0:--X-----------
              |
        qr1:--X---.--[H]--
                  |
        qr2:-----(+)------
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.cx(qr[1], qr[2])
        expected.h(qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap(self):
        """A far swap that affects coming CXs.
        qr0:--(+)---.--
               |    |
        qr1:---|----|--
               |    |
        qr2:---|----|--
               |    |
        qr3:---.---(+)-

        CouplingMap map: [0]--[1]--[2]--[3]

        qr0:--X--------------
              |
        qr1:--X--X-----------
                 |
        qr2:-----X--(+)---.--
                     |    |
        qr3:---------.---(+)-

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[3], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_front(self):
        """A far swap with a gate in the front.
        q0:------(+)--
                  |
        q1:-------|---
                  |
        q2:-------|---
                  |
        q3:--[H]--.---

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-----------(+)--
                       |
        q1:---------X--.---
                    |
        q2:------X--X------
                 |
        q3:-[H]--X---------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_back(self):
        """A far swap with a gate in the back.
        q0:--(+)------
              |
        q1:---|-------
              |
        q2:---|-------
              |
        q3:---.--[H]--

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-------(+)------
                   |
        q1:-----X--.--[H]--
                |
        q2:--X--X----------
             |
        q3:--X-------------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_middle(self):
        """A far swap with a gate in the middle.
        q0:--(+)-------.--
              |        |
        q1:---|--------|--
              |
        q2:---|--------|--
              |        |
        q3:---.--[H]--(+)-

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-------(+)-------.---
                   |        |
        q1:-----X--.--[H]--(+)--
                |
        q2:--X--X---------------
             |
        q3:--X------------------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_controlflow_pre_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(cqc, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_pre_if_else_route_post_x(self):
        """test swap with if else controlflow construct; pre-cx and post x"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg)
        qc.x(0)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(cqc, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_post_if_else_route(self):
        """test swap with if else controlflow construct; post cx"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.cx(0, 2)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(cqc, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_intra_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_pre_intra_if_else(self):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_pre_intra_post_if_else(self):
        """test swap with if else controlflow construct; cx before, in, and after if
        statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.h(3)
        qc.cx(3, 0)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_no_layout_change(self):
        """test controlflow with no layout change needed"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    @data(1, 2, 3, 4, 5)
    def test_controlflow_for_loop(self, nloops):
        """test for loop"""
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        for_body = QuantumCircuit(qreg, creg)
        for_body.cx(0, 2)
        loop_parameter = None
        qc.for_loop(range(nloops), loop_parameter, for_body, qreg, creg)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_while_loop(self):
        """test while loop"""
        from qiskit.circuit.library.standard_gates import CCXGate

        shots = 100
        num_qubits = 4
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(len(qreg))
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        while_body = QuantumCircuit(qreg, creg)
        while_body.reset(qreg[2:])
        while_body.h(qreg[2:])
        while_body.compose(CCXGate().definition, [2, 3, 0], inplace=True)
        while_body.measure(qreg[0], creg[0])
        qc.while_loop((creg, 0), while_body, qc.qubits, qc.clbits)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)
        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=shots, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=shots, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_nested_inner_cnot(self):
        """test swap in nested if else controlflow construct; swap in inner"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(0)

        for_body = QuantumCircuit(qreg, creg)
        for_body.delay(10, 0)
        for_body.cx(1, 3)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, creg)

        false_body = QuantumCircuit(qreg, creg)
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_nested_outer_cnot(self):
        """test swap with nested if else controlflow construct; swap in outer"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        true_body.x(0)

        for_body = QuantumCircuit(qreg, creg)
        for_body.delay(10, 0)
        for_body.cx(1, 3)
        loop_parameter = None
        true_body.for_loop(range(5), loop_parameter, for_body, qreg, creg)

        false_body = QuantumCircuit(qreg, creg)
        # false_body.cx(0, 4)
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096, seed_simulator=10).result().get_counts()
        out_results = sim.run(cqc, shots=4096, seed_simulator=11).result().get_counts()
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)

    def test_controlflow_continue(self):
        """test controlflow continue"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        for_body = QuantumCircuit(qreg, creg)
        for_body.cx(0, 2)
        for_body.continue_loop()

        loop_parameter = None
        qc.for_loop(range(3), loop_parameter, for_body, qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        efor_body = QuantumCircuit(qreg, creg)
        efor_body.swap(0, 1)
        efor_body.cx(1, 2)
        efor_body.continue_loop()
        efor_body.swap(0, 1)
        expected.for_loop(range(3), loop_parameter, efor_body, qreg, creg)
        self.assertEqual(cqc, expected)


if __name__ == "__main__":
    unittest.main()
