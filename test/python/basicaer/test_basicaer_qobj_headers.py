# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Tests for all BasicAer  simulators."""

from qiskit import BasicAer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import compile  # pylint: disable=redefined-builtin
from qiskit.qobj import QobjHeader

from ..common import QiskitTestCase


class TestBasicAerQobj(QiskitTestCase):
    """Tests for all the Terra simulators."""

    def setUp(self):
        super().setUp()

        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self.qc1 = QuantumCircuit(qr, cr, name='circuit0')
        self.qc1.h(qr[0])

    def test_qobj_headers_in_result(self):
        """Test that the qobj headers are passed onto the results."""
        custom_qobj_header = {'x': 1, 'y': [1, 2, 3], 'z': {'a': 4}}

        for backend in BasicAer.backends():
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

    def test_job_qobj(self):
        """Test job.qobj()."""
        for backend in BasicAer.backends():
            with self.subTest(backend=backend):
                qobj = compile(self.qc1, backend)

                job = backend.run(qobj)
                self.assertEqual(job.qobj(), qobj)
