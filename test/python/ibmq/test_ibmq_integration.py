# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin


"""IBMQ provider integration tests (compile and run)."""

import unittest

from qiskit import IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import transpile
from qiskit import compile, execute
from qiskit.result import Result
from qiskit.providers.ibmq import least_busy
from qiskit.test import QiskitTestCase, requires_qe_access


class TestCompiler(QiskitTestCase):
    """Qiskit Compiler Tests."""

    seed = 42

    def setUp(self):
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self._qc1 = QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self._qc2.x(qr[0])
        self._qc2.measure(qr[0], cr[0])

    @requires_qe_access
    def test_ibmq_result_fields(self, qe_token, qe_url):
        """Test components of a result from a remote simulator."""
        IBMQ.enable_account(qe_token, qe_url)
        remote_backend = IBMQ.get_backend(local=False, simulator=True)
        remote_result = execute(self._qc1, remote_backend).result()
        self.assertEqual(remote_result.backend_name, remote_backend.name())
        self.assertIsInstance(remote_result.job_id, str)
        self.assertEqual(remote_result.status, 'COMPLETED')
        self.assertEqual(remote_result.results[0].status, 'DONE')

    @requires_qe_access
    def test_compile_remote(self, qe_token, qe_url):
        """Test Compiler remote."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(IBMQ.backends())

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile(qc, backend)
        self.assertIsInstance(circuits, QuantumCircuit)

    @requires_qe_access
    def test_compile_two_remote(self, qe_token, qe_url):
        """Test Compiler remote on two circuits."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(IBMQ.backends())

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        circuits = transpile([qc, qc_extra], backend)
        self.assertIsInstance(circuits[0], QuantumCircuit)
        self.assertIsInstance(circuits[1], QuantumCircuit)

    @requires_qe_access
    def test_compile_run_remote(self, qe_token, qe_url):
        """Test Compiler and run remote."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qobj = compile(qc, backend, seed=TestCompiler.seed)
        job = backend.run(qobj)
        result = job.result(timeout=20)
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, qe_token, qe_url):
        """Test Compiler and run two circuits."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = compile([qc, qc_extra], backend, seed=TestCompiler.seed)
        job = backend.run(qobj)
        result = job.result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, qe_token, qe_url):
        """Test Execute remote."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        job = execute(qc, backend, seed=TestCompiler.seed)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_execute_two_remote(self, qe_token, qe_url):
        """Test execute two remote."""
        IBMQ.enable_account(qe_token, qe_url)
        backend = IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend, seed=TestCompiler.seed)
        results = job.result()
        self.assertIsInstance(results, Result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
