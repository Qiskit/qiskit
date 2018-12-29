# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin


"""Compiler Test."""

import unittest

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import transpile
from qiskit import compile
from qiskit.result import Result
from qiskit import execute
from qiskit.providers.ibmq import least_busy
from ..common import QiskitTestCase
from ..common import requires_qe_access


class TestCompiler(QiskitTestCase):
    """Qiskit Compiler Tests."""

    seed = 42

    @requires_qe_access
    def test_compile_remote(self, qe_token, qe_url):
        """Test Compiler remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(qiskit.IBMQ.backends())

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
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(qiskit.IBMQ.backends())

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
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

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
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

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
        """Test Execute remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

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
        """Test execute two remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

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
