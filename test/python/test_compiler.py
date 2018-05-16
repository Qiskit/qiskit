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
import numpy as np
import scipy.linalg as la
import qiskit
import qiskit._compiler
from qiskit import Result
from qiskit.wrapper import get_backend, execute
from qiskit.backends.ibmq import IBMQProvider
from qiskit.mapper import two_qubit_kak

from .common import requires_qe_access, QiskitTestCase


def lowest_pending_jobs(list_of_backends):
    """Returns the backend with lowest pending jobs."""
    by_pending_jobs = sorted(list_of_backends,
                             key=lambda x: x.status['pending_jobs'])
    return by_pending_jobs[0]


class TestCompiler(QiskitTestCase):
    """QISKit Compiler Tests."""

    def test_compile(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit._compiler.compile(qc, backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    def test_compile_two(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    def test_compile_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit._compiler.compile(qc, backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
        self.assertIsInstance(result, Result)

    def test_compile_two_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
        self.assertIsInstance(result, Result)

    def test_execute(self):
        """Test Execute.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        job = qiskit.wrapper.execute(qc, backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    def test_execute_two(self):
        """Test execute two.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_compile_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}))

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit._compiler.compile(qc, backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_two_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}))

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_run_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile(qc, backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, QE_TOKEN, QE_URL):
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)
        job = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                            preformatted=True))
        result = job.result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, QE_TOKEN, QE_URL):
        """Test Execute remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        job = execute(qc, backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_execute_two_remote(self, QE_TOKEN, QE_URL):
        """Test execute two remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_mapping_correction(self, QE_TOKEN, QE_URL):
        """Test mapping works in previous failed case.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.get_backend('ibmqx5')
        circuit = build_model_circuits(n=11, depth=2, num_circ=1)
        try:
            qobj = qiskit._compiler.compile(circuit, backend)
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(True)


# Helper functions for QV
def random_SU(n):
    """Return an n x n Haar distributed unitary matrix,
    using QR-decomposition on a random n x n.
    """
    X = (np.random.randn(n, n) + 1j * np.random.randn(n, n))
    Q, _ = la.qr(X)           # Q is a unitary matrix
    Q /= pow(la.det(Q), 1/n)  # make Q a special unitary
    return Q


def build_model_circuits(n, depth, num_circ=1):
    """Create a quantum program containing model circuits.
    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.
    Args:
        n (int): number of qubits
        depth (int): ideal depth of each model circuit (over SU(4))
        num_circ (int): number of model circuits to construct
    Returns:
        list(QuantumCircuit): list of quantum volume circuits
    """
    # Create quantum/classical registers of size n
    q = qiskit.QuantumRegister(name='qr', size=n)
    c = qiskit.ClassicalRegister(name='qc', size=n)
    # For each sample number, build the model circuits
    circuits = []
    for _ in range(num_circ):
        # Initialize empty circuit
        circuit = qiskit.QuantumCircuit(q, c)
        # For each layer
        for _ in range(depth):
            # Generate uniformly random permutation Pj of [0...n-1]
            perm = np.random.permutation(n)
            # For each consecutive pair in Pj, generate Haar random SU(4)
            # Decompose each SU(4) into CNOT + SU(2) and add to Ci
            for k in range(int(np.floor(n/2))):
                qubits = [int(perm[2*k]), int(perm[2*k+1])]
                SU = random_SU(4)
                decomposed_SU = two_qubit_kak(SU)
                for gate in decomposed_SU:
                    i0 = qubits[gate["args"][0]]
                    if gate["name"] == "cx":
                        i1 = qubits[gate["args"][1]]
                        circuit.cx(q[i0], q[i1])
                    elif gate["name"] == "u1":
                        circuit.u1(gate["params"][2], q[i0])
                    elif gate["name"] == "u2":
                        circuit.u2(gate["params"][1], gate["params"][2], q[i0])
                    elif gate["name"] == "u3":
                        circuit.u3(gate["params"][0], gate["params"][1],
                                   gate["params"][2], q[i0])
                    elif gate["name"] == "id":
                        pass
        # Barrier before measurement to prevent reordering, then measure
        circuit.barrier(q)
        circuit.measure(q, c)
        # Save sample circuit
        circuits.append(circuit)
    return circuits


if __name__ == '__main__':
    unittest.main(verbosity=2)
