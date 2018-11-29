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
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister, compile)

from qiskit import IBMQ, Aer
from qiskit.qasm import pi
from ..common import requires_qe_access, JobTestCase, slow_test


class TestIBMQQobj(JobTestCase):
    """Qiskit backend qobj test. Compares remote simulator as
       configured in environment variables 'IBMQ_QOBJ_DEVICE',
       'IBMQ_TOKEN' and 'IBMQ_QOBJ_URL' against local simulator
       'local_qasm_simulator' as ground truth.
    """

    def setUp(self):
        super().setUp()
        self._testing_device = os.getenv('IBMQ_QOBJ_DEVICE', None)
        self._qe_token = os.getenv('IBMQ_TOKEN', None)
        self._qe_url = os.getenv('IBMQ_QOBJ_URL')

        if not self._testing_device or not self._qe_token or not self._qe_url:
            self.skipTest("No credentials or testing device available for "
                          "testing Qobj capabilities.")

        IBMQ.enable_account(self._qe_token, self._qe_url)
        self._local_backend = Aer.get_backend('qasm_simulator_py')
        self._remote_backend = IBMQ.get_backend(self._testing_device)
        self.log.info('Remote backend: %s', self._remote_backend.name())
        self.log.info('Local backend: %s', self._local_backend.name())

    @slow_test
    @requires_qe_access
    def test_operational(self):
        """Test if backend is operational.
        """
        self.assertTrue(self._remote_backend.status().operational)

    @slow_test
    @requires_qe_access
    def test_allow_qobj(self):
        """Test if backend support Qobj.
        """
        self.assertTrue(getattr(self._remote_backend.configuration(),
                                'allow_q_object', False))

    @slow_test
    @requires_qe_access
    def test_one_qubit_no_operation(self):
        """Test one circuit, one register, in-order readout.
        """
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        circ = QuantumCircuit(qr, cr)
        circ.measure(qr[0], cr[0])

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_one_qubit_operation(self):
        """Test one circuit, one register, in-order readout.
        """
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.measure(qr[0], cr[0])

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_simple_circuit(self):
        """Test one circuit, one register, in-order readout.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_readout_order(self):
        """Test one circuit, one register, out-of-order readout.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[2])
        circ.measure(qr[1], cr[0])
        circ.measure(qr[2], cr[1])
        circ.measure(qr[3], cr[3])

        qobj_remote = compile(circ, self._remote_backend)
        qobj_local = compile(circ, self._local_backend)
        result_remote = self._remote_backend.run(qobj_remote).result()
        result_local = self._local_backend.run(qobj_local).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_multi_register(self):
        """Test one circuit, two registers, out-of-order readout.
        """
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

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_multi_circuit(self):
        """Test two circuits, two registers, out-of-order readout.
        """
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

        qobj = compile([circ1, circ2], self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ1),
                                   result_local.get_counts(circ1), delta=50)
        self.assertDictAlmostEqual(result_remote.get_counts(circ2),
                                   result_local.get_counts(circ2), delta=50)

    @slow_test
    @requires_qe_access
    def test_conditional_operation(self):
        """Test conditional operation.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[0])
        circ.x(qr[0]).c_if(cr, 1)

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)

    @slow_test
    @requires_qe_access
    def test_atlantic_circuit(self):
        """Test Atlantis deterministic ry operation.
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circ = QuantumCircuit(qr, cr)
        circ.ry(pi, qr[0])
        circ.ry(pi, qr[2])
        circ.measure(qr, cr)

        qobj = compile(circ, self._remote_backend)
        result_remote = self._remote_backend.run(qobj).result()
        result_local = self._local_backend.run(qobj).result()
        self.assertDictAlmostEqual(result_remote.get_counts(circ),
                                   result_local.get_counts(circ), delta=50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
