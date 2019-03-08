# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Tests for checking qiskit interfaces to simulators."""

import unittest
import qiskit
import qiskit.extensions.simulator
from qiskit.quantum_info import state_fidelity
from qiskit import execute
from qiskit.test import QiskitTestCase, requires_aer_provider


@requires_aer_provider
class TestCrossSimulation(QiskitTestCase):
    """Test output consistency across simulators (from built-in and legacy simulators & IBMQ)
    """
    _desired_fidelity = 0.99

    def test_statevector(self):
        """statevector from a bell state"""
        qr = qiskit.QuantumRegister(2)
        circuit = qiskit.QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])

        sim_cpp = qiskit.providers.aer.StatevectorSimulator()
        sim_py = qiskit.providers.basicaer.StatevectorSimulatorPy()
        result_cpp = execute(circuit, sim_cpp).result()
        result_py = execute(circuit, sim_py).result()
        statevector_cpp = result_cpp.get_statevector()
        statevector_py = result_py.get_statevector()
        fidelity = state_fidelity(statevector_cpp, statevector_py)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "cpp vs. py statevector has low fidelity{0:.2g}.".format(fidelity))

    def test_qasm(self):
        """counts from a GHZ state"""
        qr = qiskit.QuantumRegister(3)
        cr = qiskit.ClassicalRegister(3)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr, cr)

        sim_cpp = qiskit.providers.aer.QasmSimulator()
        sim_py = qiskit.providers.basicaer.QasmSimulatorPy()
        shots = 2000
        result_cpp = execute(circuit, sim_cpp, shots=shots).result()
        result_py = execute(circuit, sim_py, shots=shots).result()
        counts_cpp = result_cpp.get_counts()
        counts_py = result_py.get_counts()
        self.assertDictAlmostEqual(counts_cpp, counts_py, shots*0.08)

    def test_qasm_reset_measure(self):
        """counts from a qasm program with measure and reset in the middle"""
        qr = qiskit.QuantumRegister(3)
        cr = qiskit.ClassicalRegister(3)
        circuit = qiskit.QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.reset(qr[0])
        circuit.cx(qr[1], qr[2])
        circuit.t(qr)
        circuit.measure(qr[1], cr[1])
        circuit.h(qr[2])
        circuit.measure(qr[2], cr[2])

        sim_cpp = qiskit.providers.aer.QasmSimulator()
        sim_py = qiskit.providers.basicaer.QasmSimulatorPy()
        shots = 1000
        result_cpp = execute(circuit, sim_cpp, shots=shots, seed=1).result()
        result_py = execute(circuit, sim_py, shots=shots, seed=1).result()
        counts_cpp = result_cpp.get_counts()
        counts_py = result_py.get_counts()
        self.assertDictAlmostEqual(counts_cpp, counts_py, shots * 0.06)


if __name__ == '__main__':
    unittest.main(verbosity=2)
