# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except
# pylint: disable=redefined-builtin
# pylint: disable=too-many-function-args

"""IBMQ Remote Backend Qobj Tests"""

import os
import unittest
import functools
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister, compile)
from qiskit import IBMQ, Aer
from qiskit.qasm import pi

from ..common import require_multiple_credentials, JobTestCase, slow_test

# Timeout duration
TIMEOUT = int(os.getenv("IBMQ_TESTS_TIMEOUT", 10))
BLACKLIST = ('ibmq_4_atlantis',)
ALWAYS_BLACKLISTED = ('ibmqx_hpc_qasm_simulator',)


def per_non_blacklisted_backend(*blacklist):
    """ Test Qobj support on all non-blacklisted backends claiming to support Qobj.

    Args:
        blacklist (list): List of backend string names to skip.

    Returns:
        func: Decorator.
    """
    blacklist = blacklist+ALWAYS_BLACKLISTED

    def per_qobj_backend_decorator(test):
        @require_multiple_credentials
        @functools.wraps(test)
        def _wrapper(self, *args, credentials=[], **kwargs):
            for qe_token, qe_url in credentials:
                IBMQ.enable_account(qe_token, qe_url)
            for backend in IBMQ.backends():
                config = backend.configuration()
                if config['allow_q_object'] and backend.name() not in blacklist:
                    with self.subTest(backend=backend):
                        backend_test = test if config['simulator'] else slow_test(test)
                        backend_test(self, backend, *args, **kwargs)
            for qe_token, _ in credentials:
                IBMQ.disable_accounts(token=qe_token)
        return _wrapper
    return per_qobj_backend_decorator


# pylint: disable=invalid-name
per_qobj_backend = per_non_blacklisted_backend()
# pylint: disable=invalid-name
per_restricted_qobj_backend = per_non_blacklisted_backend(*BLACKLIST)


class TestBackendQobj(JobTestCase):

    def setUp(self):
        # pylint: disable=arguments-differ
        super().setUp()
        self._local_backend = Aer.get_backend('qasm_simulator_py')

    @per_qobj_backend
    def test_operational(self, remote_backend):
        """Test if backend is operational."""
        self.assertTrue(remote_backend.status()['operational'])

    @per_restricted_qobj_backend
    def test_one_qubit_no_operation(self, remote_backend):
        """Test one circuit, one register, in-order readout."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        circ = QuantumCircuit(qr, cr)
        circ.measure(qr[0], cr[0])

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_restricted_qobj_backend
    def test_one_qubit_operation(self, remote_backend):
        """Test one circuit, one register, in-order readout."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.measure(qr[0], cr[0])

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_restricted_qobj_backend
    def test_simple_circuit(self, remote_backend):
        """Test one circuit, one register, in-order readout."""
        config = remote_backend.configuration()
        n_qubits = config['n_qubits']
        if n_qubits < 4 or config.get('n_registers', n_qubits) < 4:
            self.skipTest('Backend does not have enough qubits or registers to run test.')
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_restricted_qobj_backend
    def test_readout_order(self, remote_backend):
        """Test one circuit, one register, out-of-order readout."""
        config = remote_backend.configuration()
        n_qubits = config['n_qubits']
        if n_qubits < 4 or config.get('n_registers', n_qubits) < 4:
            self.skipTest('Backend does not have enough qubits or registers to run test.')
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[2])
        circ.measure(qr[1], cr[0])
        circ.measure(qr[2], cr[1])
        circ.measure(qr[3], cr[3])

        qobj_remote = compile(circ, remote_backend)
        qobj_local = compile(circ, self._local_backend)
        result_remote = remote_backend.run(qobj_remote).result()
        result_local = self._local_backend.run(qobj_local).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_restricted_qobj_backend
    def test_multi_register(self, remote_backend):
        """Test one circuit, two registers, out-of-order readout."""
        config = remote_backend.configuration()
        n_qubits = config['n_qubits']
        if n_qubits < 4 or config.get('n_registers', n_qubits) < 4:
            self.skipTest('Backend does not have enough qubits or registers to run test.')
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(1)
        circ = QuantumCircuit(qr1, qr2, cr1, cr2)
        circ.h(qr1[0])
        circ.cx(qr1[0], qr2[1])
        circ.h(qr2[0])
        circ.cx(qr2[0], qr1[1])
        circ.x(qr1[1])
        circ.measure(qr1[0], cr2[0])
        circ.measure(qr1[1], cr1[0])
        circ.measure(qr1[1], cr2[0])
        circ.measure(qr1[1], cr1[2])
        circ.measure(qr2[0], cr1[2])
        circ.measure(qr2[1], cr1[1])

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_restricted_qobj_backend
    def test_multi_circuit(self, remote_backend):
        """Test two circuits, two registers, out-of-order readout."""
        config = remote_backend.configuration()
        n_qubits = config['n_qubits']
        if n_qubits < 4 or config.get('n_registers', n_qubits) < 4:
            self.skipTest('Backend does not have enough qubits or registers to run test.')
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(1)
        circ1 = QuantumCircuit(qr1, qr2, cr1, cr2)
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr2[1])
        circ1.h(qr2[0])
        circ1.cx(qr2[0], qr1[1])
        circ1.x(qr1[1])
        circ1.measure(qr1[0], cr2[0])
        circ1.measure(qr1[1], cr1[0])
        circ1.measure(qr1[0], cr2[0])
        circ1.measure(qr1[1], cr1[2])
        circ1.measure(qr2[0], cr1[2])
        circ1.measure(qr2[1], cr1[1])
        circ2 = QuantumCircuit(qr1, qr2, cr1)
        circ2.h(qr1[0])
        circ2.cx(qr1[0], qr1[1])
        circ2.h(qr2[1])
        circ2.cx(qr2[1], qr1[1])
        circ2.measure(qr1[0], cr1[0])
        circ2.measure(qr1[1], cr1[1])
        circ2.measure(qr1[0], cr1[2])
        circ2.measure(qr2[1], cr1[2])

        qobj = compile([circ1, circ2], remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ1),
                                   result_local.get_counts(circ1), delta=100)
        self.assertDictAlmostEqual(result_remote.get_counts(circ2),
                                   result_local.get_counts(circ2), delta=100)

    @per_restricted_qobj_backend
    def test_conditional_operation(self, remote_backend):
        """Test conditional operation.
        """
        config = remote_backend.configuration()
        if not config.get('conditional', False):
            self.skipTest('Backend does not support conditional tests')
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.measure(qr[0], cr[0])
        circ.x(qr[0]).c_if(cr, 1)

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)

    @per_qobj_backend
    def test_ry_circuit(self, remote_backend):
        """Test Atlantis staging device deterministic ry operation."""
        config = remote_backend.configuration()
        n_qubits = config['n_qubits']
        if n_qubits < 3 or config.get('n_registers', n_qubits) < 3:
            self.skipTest('Backend does not have enough qubits or registers to run test.')
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circ = QuantumCircuit(qr, cr)
        circ.ry(pi, qr[0])
        circ.ry(pi, qr[2])
        circ.measure(qr, cr)

        qobj = compile(circ, remote_backend)
        result_remote = remote_backend.run(qobj).result(timeout=TIMEOUT)
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
