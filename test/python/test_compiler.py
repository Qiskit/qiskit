# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Compiler Test."""

import unittest
import qiskit
from qiskit import Result
from qiskit.backends._qasmsimulator import QasmSimulator
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider

from .common import requires_qe_access, QiskitTestCase


def lowest_pending_jobs(list_of_backend_names, provider):
    """Returns the backend with lowest pending jobs."""
    list_of_backends = [provider.get_backend(backend) for backend
                        in list_of_backend_names]
    by_pending_jobs = sorted(list_of_backends,
                             key=lambda x: x.status['pending_jobs'])
    return by_pending_jobs[0]


class TestCompiler(QiskitTestCase):
    """QISKit Compiler Tests."""

    def test_compile(self):
        """Test Compiler.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit.compile(qc, my_backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    def test_compile_two(self):
        """Test Compiler.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit.compile([qc, qc_extra], my_backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    def test_compile_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit.compile(qc, my_backend)
        result = my_backend.run(qiskit.QuantumJob(qobj, preformatted=True))
        self.assertIsInstance(result, Result)

    def test_compile_two_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit.compile([qc, qc_extra], my_backend)
        result = my_backend.run(qiskit.QuantumJob(qobj, preformatted=True))
        self.assertIsInstance(result, Result)

    def test_execute(self):
        """Test Execute.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        results = qiskit.execute(qc, my_backend)

        self.assertIsInstance(results, Result)

    def test_execute_two(self):
        """Test execute two.

        If all correct some should exists.
        """
        my_backend = QasmSimulator()

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        results = qiskit.execute([qc, qc_extra], my_backend)

        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_compile_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}),
            provider)

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit.compile(qc, my_backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_two_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}),
            provider)

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit.compile([qc, qc_extra], my_backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_run_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = provider.get_backend('ibmqx_qasm_simulator')
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit.compile(qc, my_backend)
        result = my_backend.run(qiskit.QuantumJob(qobj, preformatted=True))
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = provider.get_backend('ibmqx_qasm_simulator')
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit.compile([qc, qc_extra], my_backend)
        result = my_backend.run(qiskit.QuantumJob(qobj, preformatted=True))
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, QE_TOKEN, QE_URL):
        """Test Execute remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = provider.get_backend('ibmqx_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        results = qiskit.execute(qc, my_backend)
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_execute_two_remote(self, QE_TOKEN, QE_URL):
        """Test execute two remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        my_backend = provider.get_backend('ibmqx_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        results = qiskit.execute([qc, qc_extra], my_backend)

        self.assertIsInstance(results, Result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
