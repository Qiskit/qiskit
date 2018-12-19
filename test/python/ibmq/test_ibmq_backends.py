# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Tests for all IBMQ backends."""

from qiskit import IBMQ, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import compile  # pylint: disable=redefined-builtin
from qiskit.qobj import QobjHeader

from ..common import QiskitTestCase, requires_qe_access, slow_test


class TestIBMQBackends(QiskitTestCase):
    """Tests for all the IBMQ backends."""

    def setUp(self):
        super().setUp()

        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self.qc1 = QuantumCircuit(qr, cr, name='circuit0')
        self.qc1.h(qr[0])
        self.qc1.measure(qr, cr)

    @requires_qe_access
    def test_qobj_headers_in_result_sims(self, qe_token, qe_url):
        """Test that the qobj headers are passed onto the results for sims."""
        IBMQ.enable_account(qe_token, qe_url)
        backends = IBMQ.backends(simulator=True)

        custom_qobj_header = {'x': 1, 'y': [1, 2, 3], 'z': {'a': 4}}

        for backend in backends:
            with self.subTest(backend=backend):
                qobj = compile(self.qc1, backend)

                # Update the Qobj header.
                qobj.header = QobjHeader.from_dict(custom_qobj_header)
                # Update the Qobj.experiment header.
                qobj.experiments[0].header.some_field = 'extra info'

                result = backend.run(qobj).result()
                self.assertEqual(result.header.to_dict(), custom_qobj_header)
                self.assertEqual(result.results[0].header.some_field,
                                 'extra info')

    @slow_test
    @requires_qe_access
    def test_qobj_headers_in_result_devices(self, qe_token, qe_url):
        """Test that the qobj headers are passed onto the results for devices."""
        IBMQ.enable_account(qe_token, qe_url)
        backends = IBMQ.backends(simulator=False)

        custom_qobj_header = {'x': 1, 'y': [1, 2, 3], 'z': {'a': 4}}

        for backend in backends:
            with self.subTest(backend=backend):
                qobj = compile(self.qc1, backend)

                # Update the Qobj header.
                qobj.header = QobjHeader.from_dict(custom_qobj_header)
                # Update the Qobj.experiment header.
                qobj.experiments[0].header.some_field = 'extra info'

                result = backend.run(qobj).result()
                self.assertEqual(result.header.to_dict(), custom_qobj_header)
                self.assertEqual(result.results[0].header.some_field,
                                 'extra info')
