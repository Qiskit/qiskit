# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,unused-import

"""Tests for checking qiskit interfaces to simulators."""

import unittest
import qiskit
import qiskit.extensions.simulator
from qiskit.tools.qi.qi import state_fidelity
from qiskit.wrapper import available_backends, register, execute, get_backend
from .common import requires_qe_access, QiskitTestCase, requires_cpp_simulator


@requires_cpp_simulator
class TestCrossSimulation(QiskitTestCase):
    """Test output consistency across simulators.
    """
    _desired_fidelity = 0.99

    def test_statevector(self):
        """statevector from a bell state"""
        q = qiskit.QuantumRegister(2)
        circ = qiskit.QuantumCircuit(q)
        circ.h(q[0])
        circ.cx(q[0], q[1])

        sim_cpp = 'local_statevector_simulator_cpp'
        sim_py = 'local_statevector_simulator_py'
        result_cpp = execute(circ, sim_cpp).result()
        result_py = execute(circ, sim_py).result()
        statevector_cpp = result_cpp.get_statevector()
        statevector_py = result_py.get_statevector()
        fidelity = state_fidelity(statevector_cpp, statevector_py)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "cpp vs. py statevector has low fidelity{0:.2g}.".format(fidelity))

    @requires_qe_access
    def test_qasm(self, QE_TOKEN, QE_URL):
        """counts from a GHZ state"""
        register(QE_TOKEN, QE_URL)
        q = qiskit.QuantumRegister(3)
        c = qiskit.ClassicalRegister(3)
        circ = qiskit.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.cx(q[1], q[2])
        circ.measure(q, c)

        sim_cpp = 'local_qasm_simulator_cpp'
        sim_py = 'local_qasm_simulator_py'
        sim_ibmq = 'ibmq_qasm_simulator'
        shots = 2000
        result_cpp = execute(circ, sim_cpp, shots=shots).result()
        result_py = execute(circ, sim_py, shots=shots).result()
        result_ibmq = execute(circ, sim_ibmq, shots=shots).result()
        counts_cpp = result_cpp.get_counts()
        counts_py = result_py.get_counts()
        counts_ibmq = result_ibmq.get_counts()
        self.assertDictAlmostEqual(counts_cpp, counts_py, shots*0.05)
        self.assertDictAlmostEqual(counts_py, counts_ibmq, shots*0.05)

    def test_qasm_snapshot(self):
        """snapshot a circuit at multiple places"""
        q = qiskit.QuantumRegister(3)
        c = qiskit.ClassicalRegister(3)
        circ = qiskit.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.snapshot(1)
        circ.ccx(q[0], q[1], q[2])
        circ.snapshot(2)
        circ.reset(q)
        circ.snapshot(3)
        circ.measure(q, c)

        sim_cpp = 'local_qasm_simulator_cpp'
        sim_py = 'local_qasm_simulator_py'
        result_cpp = execute(circ, sim_cpp, shots=2).result()
        result_py = execute(circ, sim_py, shots=2).result()
        snapshots_cpp = result_cpp.get_snapshots()
        snapshots_py = result_py.get_snapshots()
        self.assertEqual(snapshots_cpp.keys(), snapshots_py.keys())
        snapshot_cpp_1 = result_cpp.get_snapshot(slot='1')
        snapshot_py_1 = result_py.get_snapshot(slot='1')
        self.assertEqual(len(snapshot_cpp_1), len(snapshot_py_1))
        fidelity = state_fidelity(snapshot_cpp_1[0], snapshot_py_1[0])
        self.assertGreater(fidelity, self._desired_fidelity)

    @requires_qe_access
    def test_qasm_reset_measure(self, QE_TOKEN, QE_URL):
        """counts from a qasm program with measure and reset in the middle"""
        register(QE_TOKEN, QE_URL)
        q = qiskit.QuantumRegister(3)
        c = qiskit.ClassicalRegister(3)
        circ = qiskit.QuantumCircuit(q, c)
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.reset(q[0])
        circ.cx(q[1], q[2])
        circ.t(q)
        circ.measure(q[1], c[1])
        circ.h(q[2])
        circ.measure(q[2], c[2])

        # TODO: bring back online simulator tests when reset/measure doesn't
        # get rejected by the api
        sim_cpp = 'local_qasm_simulator_cpp'
        sim_py = 'local_qasm_simulator_py'
        # sim_ibmq = 'ibmq_qasm_simulator'
        shots = 1000
        result_cpp = execute(circ, sim_cpp, shots=shots, seed=1).result()
        result_py = execute(circ, sim_py, shots=shots, seed=1).result()
        # result_ibmq = execute(circ, sim_ibmq, {'shots': shots}).result()
        counts_cpp = result_cpp.get_counts()
        counts_py = result_py.get_counts()
        # counts_ibmq = result_ibmq.get_counts()
        self.assertDictAlmostEqual(counts_cpp, counts_py, shots * 0.04)
        # self.assertDictAlmostEqual(counts_py, counts_ibmq, shots*0.04)


if __name__ == '__main__':
    unittest.main(verbosity=2)
