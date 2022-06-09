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
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals
from qiskit.quantum_info.analysis.distance import hellinger_distance


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
        from qiskit.converters import dag_to_circuit
        from qiskit import Aer

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])        
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
        print(in_results)
        print(out_results)
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_pre_if_else_route_post_x(self):
        """test swap with if else controlflow construct"""
        from qiskit.converters import dag_to_circuit
        from qiskit import Aer

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])        
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
        print(qc)
        print(cqc)
        print(in_results)
        print(out_results)
        self.assertEqual(set(in_results), set(out_results))
        
    def test_controlflow_post_if_else_route(self):
        """test swap with if else controlflow construct"""
        from qiskit.converters import dag_to_circuit
        from qiskit import Aer

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])        
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
        print(qc)
        print(cqc)
        print(in_results)
        print(out_results)
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_intra_if_else_route(self):
        """test swap with if else controlflow construct"""
        from qiskit.converters import dag_to_circuit
        from qiskit import Aer

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])        
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
        print(qc)
        print(cqc)
        print(qc.data[3][0].blocks[0])
        print(cqc.data[3][0].blocks[0])
        print(qc.data[3][0].blocks[1])
        print(cqc.data[3][0].blocks[1])
        print(in_results)
        print(out_results)
        distance = hellinger_distance(in_results, out_results)
        print(f'hellinger distance = {distance}')
        self.assertLess(hellinger_distance(in_results, out_results), 0.01)
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_intra_if_else_route2(self):
        """test what circuit paths should look like without actually using controlflow op"""
        from qiskit.converters import dag_to_circuit
        from qiskit import Aer
        from qiskit.transpiler.passes.routing import LayoutTransformation

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])        
        qcpre = QuantumCircuit(qreg, creg)
        qcintra0 = QuantumCircuit(qreg, creg)
        qcintra1 = QuantumCircuit(qreg, creg)
        meas = QuantumCircuit(qreg, creg)
        meas.measure(qreg, creg)
    
        qcpre.h(0)
        qcpre.x(1)
        #qcpre.measure(0, 0)
        qcintra0.cx(0, 2)
        qcintra1.cx(0, 4)

        qc0 = qcpre.compose(qcintra0)
        qc1 = qcpre.compose(qcintra1)

        #qc0.measure_all(add_bits=False)
        
        dag0 = circuit_to_dag(qc0)
        bswap0 = BasicSwap(coupling)
        cdag0 = bswap0.run(dag0)
        cqc0 = dag_to_circuit(cdag0)
        # cqc0.swap(2, 3)
        # cqc0.swap(3, 4)
        qc0.measure_all(add_bits=False)
        # cqc0.measure_all(add_bits=False)

        qc1.measure_all(add_bits=False)
        dag1 = circuit_to_dag(qc1)
        bswap1 = BasicSwap(coupling)
        cdag1 = bswap1.run(dag1)
        cqc1 = dag_to_circuit(cdag1)

        # now use measurements from bswap1 to apply to cqc0
        order = bswap1.property_set["final_layout"].reorder_bits(cqc1.qubits)
        cqc0.barrier()
        #cqc0.compose(meas, qubits=[0, 3, 1, 2, 4], inplace=True)
        new_list = list(iter(order))
        cqc0.compose(meas, qubits=new_list, inplace=True)

        expected0 = qcpre.copy()
        expected0.swap(1, 2)
        expected0.cx(2, 3)
        swap_circ = QuantumCircuit(qreg)
        swap_circ.swap(2, 3)
        #expected0.swap(2, 3)
        layout = bswap1.property_set["final_layout"]
        order = layout.reorder_bits(expected0.qubits)
        
        expected0b = expected0.compose(swap_circ, qubits=order)
        expected0b.barrier()
        expected0b.measure(qreg, creg)
        expected0_dag = circuit_to_dag(expected0b)
        #expected0_dag.compose(circuit_to_dag(meas), qubits=order)
        #expected0b = dag_to_circuit(expected0_dag)
        
        sim = Aer.get_backend("aer_simulator")
        in_results0 = sim.run(qc0, shots=4096).result().get_counts()
        in_results1 = sim.run(qc1, shots=4096).result().get_counts()        
        out_results0 = sim.run(cqc0, shots=4096).result().get_counts()
        out_results1 = sim.run(cqc1, shots=4096).result().get_counts()
        expected0_results = sim.run(expected0b, shots=4096).result().get_counts()
        print('qc0')
        print(qc0)
        print(in_results0)
        print('cqc0')
        print(cqc0)
        print(out_results0)
        print('qc1')
        print(qc1)
        print(in_results1)        
        print('cqc1')
        print(cqc1)
        print(out_results1)
        print('expected0')
        print(expected0b)
        print(expected0_results)
        xform = LayoutTransformation(coupling, bswap0.property_set["final_layout"],
                                     bswap1.property_set["final_layout"])
        cdag02 = xform.run(cdag0)
        ccirc02 = dag_to_circuit(cdag02)
        ccirc02.barrier()
        ccirc02.compose(meas, qubits=order, inplace=True)
        ccirc02_results = sim.run(ccirc02, shots=4096).result().get_counts()        
        print(ccirc02)
        print(ccirc02_results)
        breakpoint()
        self.assertEqual(set(in_results), set(out_results))
        
        
    def test_controlflow_if_else2(self):
        """test swap with if else controlflow construct"""
        from qiskit.converters import dag_to_circuit

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])

        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 2)
        qc.measure(2, 0)
        
        true_body = QuantumCircuit(qreg, creg)
        true_body.h(0)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(1)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)
        
        print(qc)
        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)
        from qiskit import Aer

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(cqc, shots=4096).result().get_counts()
        print(in_results)
        print(out_results)
        self.assertEqual(set(in_results), set(out_results))

    def test_controlflow_if_else(self):
        """test swap with if-else controlflow construct"""
        from qiskit.converters import dag_to_circuit

        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i+1) for i in range(num_qubits - 1)])

        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(0, 0)

        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.cx(1, 2)
        qc.measure_all()
        # there seems to be a difference when using the context manager to
        # create if_else op; the QuantumCircuit blocks have empty "qregs" but
        # qubits property looks correct
        # with qc.if_test((creg[0], 0)) as else_:
        #     qc.cx(0, 2)
        # with else_:
        #     qc.cx(0, 4)
        dag = circuit_to_dag(qc)
        cdag = BasicSwap(coupling).run(dag)
        cqc = dag_to_circuit(cdag)
        
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.measure(1, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(1, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.swap(1, 2)
        false_body.swap(2, 3)
        false_body.cx(3, 4)
        false_body.swap(2, 3)
        false_body.swap(1, 2)
        expected.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected_dag = circuit_to_dag(expected)
        #self.assertEqual(cqc, expected)
        #breakpoint()
        if not optionals.HAS_AER:
            return

        from qiskit import Aer

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(cqc, shots=4096).result().get_counts()
        breakpoint()
        self.assertEqual(set(in_results), set(out_results))
        

        

if __name__ == "__main__":
    unittest.main()
