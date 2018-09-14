# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class."""

import qiskit.extensions.simulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QISKitError
from qiskit.tools.qi.qi import state_fidelity
from .common import QiskitTestCase


class TestCircuit(QiskitTestCase):
    """QuantumCircuit basic tests."""

    def test_get_qregs(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        q_regs = qc.get_qregs()
        self.assertEqual(len(q_regs), 2)
        self.assertEqual(q_regs[qr1.name], qr1)
        self.assertEqual(q_regs[qr2.name], qr2)

    def test_get_cregs(self):
        """Test getting classical registers from circuit.
        """
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(3)
        qc = QuantumCircuit(cr1, cr2, cr3)
        c_regs = qc.get_cregs()
        self.assertEqual(len(c_regs), 3)
        self.assertEqual(c_regs[cr1.name], cr1)
        self.assertEqual(c_regs[cr2.name], cr2)

    def test_circuit_qasm(self):
        """Test circuit qasm() method.
        """
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.u1(0.3, qr1[0])
        qc.u2(0.2, 0.1, qr2[0])
        qc.u3(0.3, 0.2, 0.1, qr2[1])
        qc.s(qr2[1])
        qc.s(qr2[1]).inverse()
        qc.cx(qr1[0], qr2[1])
        qc.barrier(qr2)
        qc.cx(qr2[1], qr1[0])
        qc.h(qr2[1])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        qc.barrier(qr1, qr2)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr1[1];
qreg qr2[2];
creg cr[3];
u1(0.300000000000000) qr1[0];
u2(0.200000000000000,0.100000000000000) qr2[0];
u3(0.300000000000000,0.200000000000000,0.100000000000000) qr2[1];
s qr2[1];
sdg qr2[1];
cx qr1[0],qr2[1];
barrier qr2[0],qr2[1];
cx qr2[1],qr1[0];
h qr2[1];
if(cr==0) x qr2[1];
if(cr==1) y qr1[0];
if(cr==2) z qr1[0];
barrier qr1[0],qr2[0],qr2[1];
measure qr1[0] -> cr[0];
measure qr2[0] -> cr[1];
measure qr2[1] -> cr[2];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)


class TestCircuitCombineExtend(QiskitTestCase):
    """Test combining and extending of QuantumCircuits."""

    def test_combine_circuit_common(self):
        """Test combining two circuits with same registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        backend = 'local_qasm_simulator'
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_combine_circuit_different(self):
        """Test combining two circuits with different registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        qc1.x(qr)
        qc2 = QuantumCircuit(qr, cr)
        qc2.measure(qr, cr)
        new_circuit = qc1 + qc2
        backend = 'local_qasm_simulator'
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_combine_circuit_fail(self):
        """Test combining two circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QISKitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QISKitError, qc1.__add__, qc2)
        self.assertRaises(QISKitError, qc1.__add__, qcr3)

    def test_combine_circuit_extension_instructions(self):
        """Test combining circuits contining barrier, initializer, snapshot
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qc1.initialize(desired_vector, qr)
        qc1.barrier()
        qc2 = QuantumCircuit(qr, cr)
        qc2.snapshot(slot='1')
        qc2.measure(qr, cr)
        new_circuit = qc1 + qc2
        backend = 'local_qasm_simulator_py'
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()

        snapshot_vectors = result.get_snapshot()
        fidelity = state_fidelity(snapshot_vectors[0], desired_vector)
        self.assertGreater(fidelity, 0.99)

        counts = result.get_counts()
        target = {'00': shots/4, '01': shots/4, '10': shots/4, '11': shots/4}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit(self):
        """Test extending a circuit with same registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc1 += qc2
        backend = 'local_qasm_simulator'
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit_different_registers(self):
        """Test extending a circuit with different registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        qc1.x(qr)
        qc2 = QuantumCircuit(qr, cr)
        qc2.measure(qr, cr)
        qc1 += qc2
        backend = 'local_qasm_simulator'
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_extend_circuit_fail(self):
        """Test extending a circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QISKitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QISKitError, qc1.__iadd__, qc2)
        self.assertRaises(QISKitError, qc1.__iadd__, qcr3)

    def test_extend_circuit_extension_instructions(self):
        """Test extending circuits contining barrier, initializer, snapshot
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qc1.initialize(desired_vector, qr)
        qc1.barrier()
        qc2 = QuantumCircuit(qr, cr)
        qc2.snapshot(slot='1')
        qc2.measure(qr, cr)
        qc1 += qc2
        backend = 'local_qasm_simulator_py'
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()

        snapshot_vectors = result.get_snapshot('1')
        fidelity = state_fidelity(snapshot_vectors[0], desired_vector)
        self.assertGreater(fidelity, 0.99)

        counts = result.get_counts()
        target = {'00': shots/4, '01': shots/4, '10': shots/4, '11': shots/4}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)
