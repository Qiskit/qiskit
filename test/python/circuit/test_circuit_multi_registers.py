# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import
# pylint: disable=redefined-builtin

"""Test Qiskit's QuantumCircuit class for multiple registers."""

import os
import tempfile
import unittest

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile, execute
from qiskit import QiskitError
from qiskit.quantum_info import state_fidelity, process_fidelity, Pauli, basis_state
from ..common import QiskitTestCase, requires_cpp_simulator, bin_to_hex_keys


class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    @requires_cpp_simulator
    def test_circuit_multi(self):
        """Test circuit multi regs declared at start.
        """
        qreg0 = QuantumRegister(2, 'q0')
        creg0 = ClassicalRegister(2, 'c0')
        qreg1 = QuantumRegister(2, 'q1')
        creg1 = ClassicalRegister(2, 'c1')
        circ = QuantumCircuit(qreg0, qreg1)
        circ.x(qreg0[1])
        circ.x(qreg1[0])

        meas = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        meas.measure(qreg0, creg0)
        meas.measure(qreg1, creg1)

        qc = circ + meas

        backend_sim = Aer.get_backend('qasm_simulator_py')
        qobj_qc = compile(qc, backend_sim, seed_mapper=34342)
        qobj_circ = compile(circ, backend_sim, seed_mapper=3438)

        result = backend_sim.run(qobj_qc).result()
        counts = result.get_counts(qc)

        backend_sim = Aer.get_backend('qasm_simulator')
        result = backend_sim.run(qobj_qc).result()
        counts_py = result.get_counts(qc)

        target = bin_to_hex_keys({'01 10': 1024})

        backend_sim = Aer.get_backend('statevector_simulator')
        result = backend_sim.run(qobj_circ).result()
        state = result.get_statevector(circ)

        backend_sim = Aer.get_backend('statevector_simulator_py')
        result = backend_sim.run(qobj_circ).result()
        state_py = result.get_statevector(circ)

        backend_sim = Aer.get_backend('unitary_simulator_py')
        result = backend_sim.run(qobj_circ).result()
        unitary = result.get_unitary(circ)

        self.assertEqual(counts, target)
        self.assertEqual(counts_py, target)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state_py), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                               1.0, places=7)

    def test_circuit_multi_case2(self):
        """Test circuit multi regs declared at start.
        """
        qreg0 = QuantumRegister(2, 'q0')
        creg0 = ClassicalRegister(2, 'c0')
        qreg1 = QuantumRegister(2, 'q1')
        creg1 = ClassicalRegister(2, 'c1')
        circ2 = QuantumCircuit()
        circ2.add_register(qreg0)
        circ2.add_register(qreg1)
        circ2.x(qreg0[1])
        circ2.x(qreg1[0])

        meas2 = QuantumCircuit()
        meas2.add_register(qreg0)
        meas2.add_register(qreg1)
        meas2.add_register(creg0)
        meas2.add_register(creg1)
        meas2.measure(qreg0, creg0)
        meas2.measure(qreg1, creg1)

        qc2 = circ2 + meas2

        backend_sim = Aer.get_backend('statevector_simulator_py')
        result = execute(circ2, backend_sim).result()
        state = result.get_statevector(circ2)

        backend_sim = Aer.get_backend('qasm_simulator_py')
        result = execute(qc2, backend_sim).result()
        counts = result.get_counts(qc2)

        backend_sim = Aer.get_backend('unitary_simulator_py')
        result = execute(circ2, backend_sim).result()
        unitary = result.get_unitary(circ2)

        target = bin_to_hex_keys({'01 10': 1024})
        self.assertEqual(target, counts)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                               1.0, places=7)
