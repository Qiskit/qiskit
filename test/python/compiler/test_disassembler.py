# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assembler Test."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.compiler.assemble import assemble
from qiskit.assembler.disassemble import disassemble
from qiskit.assembler.run_config import RunConfig
from qiskit.test import QiskitTestCase
import qiskit.quantum_info as qi


class TestAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def test_disassemble_single_circuit(self):
        """Test assembling a single circuit.
        """
        qr = QuantumRegister(2, name='q')
        cr = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, cr, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)

        qobj = assemble(circ, shots=2000, memory=True)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, True)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disassemble_multiple_circuits(self):
        """Test assembling multiple circuits, all should have the same config.
        """
        qr0 = QuantumRegister(2, name='q0')
        qc0 = ClassicalRegister(2, name='c0')
        circ0 = QuantumCircuit(qr0, qc0, name='circ0')
        circ0.h(qr0[0])
        circ0.cx(qr0[0], qr0[1])
        circ0.measure(qr0, qc0)

        qr1 = QuantumRegister(3, name='q1')
        qc1 = ClassicalRegister(3, name='c1')
        circ1 = QuantumCircuit(qr1, qc1, name='circ0')
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr1[1])
        circ1.cx(qr1[0], qr1[2])
        circ1.measure(qr1, qc1)

        qobj = assemble([circ0, circ1], shots=100, memory=False, seed=6)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(run_config_out.shots, 100)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.seed, 6)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertIn(circuit, [circ0, circ1])
        self.assertEqual({}, headers)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default.
        """
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        qobj = assemble(circ)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_disassemble_isometry(self):
        """Test disassembling a circuit with an isometry.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.iso(qi.random_unitary(4).data, circ.qubits, [])
        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        # params array
        assert_allclose(circuits[0]._data[0][0].params[0], circ._data[0][0].params[0])
        # all other data
        self.assertEqual(circuits[0]._data[0][0].params[1:], circ._data[0][0].params[1:])
        self.assertEqual(circuits[0]._data[0][1:], circ._data[0][1:])
        self.assertEqual(circuits[0]._data[1:], circ._data[1:])
        self.assertEqual({}, header)

    def test_opaque_instruction(self):
        """Test the disassembler handles opaque instructions correctly."""
        opaque_inst = Instruction(name='my_inst', num_qubits=4,
                                  num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name='q')
        c = ClassicalRegister(4, name='c')
        circ = QuantumCircuit(q, c, name='circ')
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])
        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 6)
        self.assertEqual(run_config_out.memory_slots, 4)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_circuit_with_conditionals(self):
        """Verify disassemble sets conditionals correctly."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr1, cr2)
        qc.measure(qr[0], cr1)  # Measure not required for a later conditional
        qc.measure(qr[1], cr2[1])  # Measure required for a later conditional
        qc.h(qr[1]).c_if(cr2, 3)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_simple_conditional(self):
        """Verify disassemble handles a simple conditional on the only bits."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0]).c_if(cr, 1)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 1)
        self.assertEqual(run_config_out.memory_slots, 1)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_multiple_conditionals_multiple_registers(self):
        """Verify disassemble handles multiple conditionals and registers."""
        qr = QuantumRegister(3)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(5)
        cr3 = ClassicalRegister(6)
        cr4 = ClassicalRegister(1)

        qc = QuantumCircuit(qr, cr1, cr2, cr3, cr4)
        qc.x(qr[1])
        qc.h(qr)
        qc.cx(qr[1], qr[0]).c_if(cr3, 14)
        qc.ccx(qr[0], qr[2], qr[1]).c_if(cr4, 1)
        qc.h(qr).c_if(cr1, 3)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 15)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)


if __name__ == '__main__':
    unittest.main(verbosity=2)
