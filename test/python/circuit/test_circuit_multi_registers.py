# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Test Qiskit's QuantumCircuit class for multiple registers."""

import os
import tempfile
import unittest

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile, execute
from qiskit import QISKitError
from qiskit.quantum_info import state_fidelity, process_fidelity, Pauli, basis_state
from ..common import QiskitTestCase


class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_multi(self):
        """Test circuit multi regs declared at start.
        """
        qubit0 = QuantumRegister(2, 'q0')
        cbit0 = ClassicalRegister(2, 'c0')
        qubit1 = QuantumRegister(2, 'q1')
        cbit1 = ClassicalRegister(2, 'c1')
        circ = QuantumCircuit(qubit0, qubit1)
        circ.x(qubit0[1])
        circ.x(qubit1[0])

        meas = QuantumCircuit(qubit0, qubit1, cbit0, cbit1)
        meas.measure(qubit0, cbit0)
        meas.measure(qubit1, cbit1)

        qc = circ + meas

        backend_sim = Aer.get_backend('qasm_simulator_py')
        qobj_qc = compile(qc, backend_sim)
        qobj_circ = compile(circ, backend_sim)
        print(qobj_qc.header.qubit_labels)
        print(qobj_qc.header.compiled_circuit_qasm)
        for gate in qobj_qc.instructions:
                print(gate)

        result = backend_sim.run(qobj_qc).result()
        counts = result.get_counts(qc)
        print(counts)

        backend_sim = Aer.get_backend('qasm_simulator')
        result = backend_sim.run(qobj_qc).result()
        counts_py = result.get_counts(qc)

        target = {'01 10': 1024}

        backend_sim = Aer.get_backend('statevector_simulator')
        result = backend_sim.run(qobj_circ).result()
        state = result.get_statevector(circ)

        backend_sim = Aer.get_backend('statevector_simulator_py')
        result = backend_sim.run(qobj_circ).result()
        state_py = result.get_statevector(circ)

        backend_sim = Aer.get_backend('unitary_simulator')
        result = backend_sim.run(qobj_circ).result()
        unitary = result.get_unitary(circ)

        # self.assertEqual(counts, target)
        # self.assertEqual(counts_py, target)
        # self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        # self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state_py), 1.0, places=7)
        # self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
        #                       1.0, places=7)

    def test_circuit_multi_case2(self):
        """Test circuit multi regs declared at start.
        """
        qubit0 = QuantumRegister(2, 'q0')
        cbit0 = ClassicalRegister(2, 'c0')
        qubit1 = QuantumRegister(2, 'q1')
        cbit1 = ClassicalRegister(2, 'c1')
        circ2 = QuantumCircuit()
        circ2.add_register(qubit0)
        circ2.add_register(qubit1)
        circ2.x(qubit0[1])
        circ2.x(qubit1[0])

        meas2 = QuantumCircuit()
        meas2.add_register(qubit0)
        meas2.add_register(qubit1)
        meas2.add_register(cbit0)
        meas2.add_register(cbit1)
        meas2.measure(qubit0, cbit0)
        meas2.measure(qubit1, cbit1)

        qc2 = circ2 + meas2

        backend_sim = Aer.get_backend('statevector_simulator')
        result = execute(circ2, backend_sim).result()
        state = result.get_statevector(circ2)

        backend_sim = Aer.get_backend('qasm_simulator')
        result = execute(qc2, backend_sim).result()
        counts = result.get_counts(qc2)

        backend_sim = Aer.get_backend('unitary_simulator')
        result = execute(circ2, backend_sim).result()
        unitary = result.get_unitary(circ2)

        target = {'01 10': 1024}
        self.assertEqual(counts, target)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                               1.0, places=7)
