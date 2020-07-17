# -*- coding: utf-8 -*-

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

# pylint: disable=invalid-name

"""Test Qiskit's inverse gate operation."""

import unittest
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError
# pylint: disable=unused-import
from qiskit.extensions.simulator import snapshot


class TestCircuitProperties(QiskitTestCase):
    """QuantumCircuit properties tests."""

    def test_qarg_numpy_int(self):
        """Test castable to integer args for QuantumCircuit.
        """
        n = np.int64(12)
        qc1 = QuantumCircuit(n)
        self.assertEqual(qc1.num_qubits, 12)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int(self):
        """Test castable to integer cargs for QuantumCircuit.
        """
        n = np.int64(12)
        c1 = ClassicalRegister(n)
        qc1 = QuantumCircuit(c1)
        c_regs = qc1.cregs
        self.assertEqual(c_regs[0], c1)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int_2(self):
        """Test castable to integer cargs for QuantumCircuit.
        """
        qc1 = QuantumCircuit(12, np.int64(12))
        c_regs = qc1.cregs
        self.assertEqual(c_regs[0], ClassicalRegister(12, 'c'))
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_qarg_numpy_int_exception(self):
        """Test attempt to pass non-castable arg to QuantumCircuit.
        """
        self.assertRaises(CircuitError, QuantumCircuit, 'string')

    def test_circuit_depth_empty(self):
        """Test depth of empty circuity
        """
        q = QuantumRegister(5, 'q')
        qc = QuantumCircuit(q)
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_no_reg(self):
        """Test depth of no register circuits
        """
        qc = QuantumCircuit()
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_meas_only(self):
        """Test depth of measurement only
        """
        q = QuantumRegister(1, 'q')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q, c)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 1)

    def test_circuit_depth_barrier(self):
        """Make sure barriers do not add to depth
        """
        q = QuantumRegister(5, 'q')
        c = ClassicalRegister(5, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(q[4])
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[4])
        qc.cx(q[4], q[2])
        qc.cx(q[2], q[3])
        qc.barrier(q)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_simple(self):
        """Test depth for simple circuit
        """
        q = QuantumRegister(5, 'q')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[4])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[4])
        qc.cx(q[4], q[1])
        qc.measure(q[1], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_multi_reg(self):
        """Test depth for multiple registers
        """
        q1 = QuantumRegister(3, 'q1')
        q2 = QuantumRegister(2, 'q2')
        c = ClassicalRegister(5, 'c')
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_3q_gate(self):
        """Test depth for 3q gate
        """
        q1 = QuantumRegister(3, 'q1')
        q2 = QuantumRegister(2, 'q2')
        c = ClassicalRegister(5, 'c')
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.ccx(q2[1], q1[0], q2[0])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_conditionals1(self):
        """Test circuit depth for conditional gates #1.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[0], q[1])
        qc.cx(q[2], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.h(q[2]).c_if(c, 2)
        qc.h(q[3]).c_if(c, 4)
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_conditionals2(self):
        """Test circuit depth for conditional gates #2.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[0], q[1])
        qc.cx(q[2], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[0], c[0])
        qc.h(q[2]).c_if(c, 2)
        qc.h(q[3]).c_if(c, 4)
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_conditionals3(self):
        """Test circuit depth for conditional gates #3.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.cx(q[0], q[3]).c_if(c, 2)

        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 4)

    def test_circuit_depth_measurements1(self):
        """Test circuit depth for measurements #1.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 2)

    def test_circuit_depth_measurements2(self):
        """Test circuit depth for measurements #2.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[0], c[1])
        qc.measure(q[0], c[2])
        qc.measure(q[0], c[3])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_measurements3(self):
        """Test circuit depth for measurements #3.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_barriers1(self):
        """Test circuit depth for barriers #1.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers2(self):
        """Test circuit depth for barriers #2.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers3(self):
        """Test circuit depth for barriers #3.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.barrier(q)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_snap1(self):
        """Test circuit depth for snapshots #1.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.cx(0, 1)
        circ.snapshot('snap')
        circ.h(2)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_snap2(self):
        """Test circuit depth for snapshots #2.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.snapshot('snap0')
        circ.cx(0, 1)
        circ.snapshot('snap1')
        circ.h(2)
        circ.snapshot('snap2')
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_snap3(self):
        """Test circuit depth for snapshots #3.
        """
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.cx(0, 1)
        circ.snapshot('snap0')
        circ.snapshot('snap1')
        circ.h(2)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_size_empty(self):
        """Circuit.size should return 0 for an empty circuit."""
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        self.assertEqual(qc.size(), 0)

    def test_circuit_size_single_qubit_gates(self):
        """Circuit.size should increment for each added single qubit gate."""
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        self.assertEqual(qc.size(), 1)
        qc.h(q[1])
        self.assertEqual(qc.size(), 2)

    def test_circuit_size_two_qubit_gates(self):
        """Circuit.size should increment for each added two qubit gate."""
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)

        qc.cx(q[0], q[1])
        self.assertEqual(qc.size(), 1)
        qc.cx(q[2], q[3])
        self.assertEqual(qc.size(), 2)

    def test_circuit_size_ignores_barriers_snapshots(self):
        """Circuit.size should not count barriers or snapshots."""
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.cx(q[0], q[1])
        self.assertEqual(qc.size(), 2)
        qc.barrier(q)
        self.assertEqual(qc.size(), 2)
        qc.snapshot('snapshot_label')
        self.assertEqual(qc.size(), 2)

    def test_circuit_count_ops(self):
        """Test circuit count ops.
        """
        q = QuantumRegister(6, 'q')
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.x(q[1])
        qc.y(q[2:4])
        qc.z(q[3:])
        result = qc.count_ops()

        expected = dict([('h', 6), ('z', 3), ('y', 2), ('x', 1)])

        self.assertIsInstance(result, dict)
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates(self):
        """Test num_nonlocal_gates.
        """
        q = QuantumRegister(6, 'q')
        c = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q)
        qc.x(q[1])
        qc.cry(0.1, q[2], q[4])
        qc.z(q[3:])
        qc.cswap(q[1], q[2], q[3])
        qc.iswap(q[0], q[4]).c_if(c, 2)
        result = qc.num_nonlocal_gates()
        expected = 3
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates_no_instruction(self):
        """Verify num_nunlocal_gates does not include barriers.
        """
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4500
        n = 3
        qc = QuantumCircuit(n)
        qc.h(range(n))

        qc.barrier()

        self.assertEqual(qc.num_nonlocal_gates(), 0)

    def test_circuit_connected_components_empty(self):
        """Verify num_connected_components is width for empty
        """
        q = QuantumRegister(7, 'q')
        qc = QuantumCircuit(q)
        self.assertEqual(7, qc.num_connected_components())

    def test_circuit_connected_components_multi_reg(self):
        """Test tensor factors works over multi registers
        """
        q1 = QuantumRegister(3, 'q1')
        q2 = QuantumRegister(2, 'q2')
        qc = QuantumCircuit(q1, q2)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_connected_components_multi_reg2(self):
        """Test tensor factors works over multi registers #2.
        """
        q1 = QuantumRegister(3, 'q1')
        q2 = QuantumRegister(2, 'q2')
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[1])
        qc.cx(q2[0], q1[2])
        qc.cx(q1[1], q2[0])
        self.assertEqual(qc.num_connected_components(), 2)

    def test_circuit_connected_components_disconnected(self):
        """Test tensor factors works with 2q subspaces.
        """
        q1 = QuantumRegister(5, 'q1')
        q2 = QuantumRegister(5, 'q2')
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[4])
        qc.cx(q1[1], q2[3])
        qc.cx(q1[2], q2[2])
        qc.cx(q1[3], q2[1])
        qc.cx(q1[4], q2[0])
        self.assertEqual(qc.num_connected_components(), 5)

    def test_circuit_connected_components_with_clbits(self):
        """Test tensor components with classical register.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 4)

    def test_circuit_connected_components_with_cond(self):
        """Test tensor components with conditional gate.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.cx(q[0], q[3]).c_if(c, 2)
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_unitary_factors1(self):
        """Test unitary factors empty circuit.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors2(self):
        """Test unitary factors multi qregs
        """
        q1 = QuantumRegister(2, 'q1')
        q2 = QuantumRegister(2, 'q2')
        c = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(q1, q2, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors3(self):
        """Test unitary factors measurements and conditionals.
        """
        size = 4
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[2])
        qc.cx(q[0], q[3]).c_if(c, 2)
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_unitary_factors(), 2)

    def test_circuit_unitary_factors4(self):
        """Test unitary factors measurements go to same cbit.
        """
        size = 5
        q = QuantumRegister(size, 'q')
        c = ClassicalRegister(size, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.num_unitary_factors(), 5)

    def test_num_qubits_qubitless_circuit(self):
        """Check output in absence of qubits.
        """
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(c_reg)
        self.assertEqual(circ.num_qubits, 0)

    def test_num_qubits_qubitfull_circuit(self):
        """Check output in presence of qubits
        """
        q_reg = QuantumRegister(4)
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(q_reg, c_reg)
        self.assertEqual(circ.num_qubits, 4)

    def test_num_qubits_registerless_circuit(self):
        """Check output for circuits with direct argument for qubits.
        """
        circ = QuantumCircuit(5)
        self.assertEqual(circ.num_qubits, 5)

    def test_num_qubits_multiple_register_circuit(self):
        """Check output for circuits with multiple quantum registers.
        """
        q_reg1 = QuantumRegister(5)
        q_reg2 = QuantumRegister(6)
        q_reg3 = QuantumRegister(7)
        circ = QuantumCircuit(q_reg1, q_reg2, q_reg3)
        self.assertEqual(circ.num_qubits, 18)


if __name__ == '__main__':
    unittest.main()
