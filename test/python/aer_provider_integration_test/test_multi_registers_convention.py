# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Test Qiskit's QuantumCircuit class for multiple registers."""

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile
from qiskit.quantum_info import state_fidelity, process_fidelity, Pauli, basis_state
from qiskit.test import QiskitTestCase, requires_aer_provider


class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    @requires_aer_provider
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

        backend_sim = qiskit.providers.aer.QasmSimulator()
        qobj_qc = compile(qc, backend_sim, seed_mapper=34342)
        qobj_circ = compile(circ, backend_sim, seed_mapper=3438)

        result = backend_sim.run(qobj_qc).result()
        counts = result.get_counts(qc)

        target = {'01 10': 1024}

        backend_sim = qiskit.providers.aer.StatevectorSimulator()
        result = backend_sim.run(qobj_circ).result()
        state = result.get_statevector(circ)

        backend_sim = qiskit.providers.aer.UnitarySimulator()
        result = backend_sim.run(qobj_circ).result()
        unitary = result.get_unitary(circ)

        self.assertEqual(counts, target)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                               1.0, places=7)
