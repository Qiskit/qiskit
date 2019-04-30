# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for all BasicAer  simulators."""

from qiskit import BasicAer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from qiskit.qobj import QobjHeader
from qiskit.test import QiskitTestCase


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
                new_circ = transpile(self.qc1, backend=backend)
                qobj = assemble(new_circ, shots=1024)

                # Update the Qobj header.
                qobj.header = QobjHeader.from_dict(custom_qobj_header)
                # Update the Qobj.experiment header.
                qobj.experiments[0].header.some_field = 'extra info'

                result = backend.run(qobj).result()
                self.assertEqual(result.header.to_dict(), custom_qobj_header)
                self.assertEqual(result.results[0].header.some_field, 'extra info')

    def test_job_qobj(self):
        """Test job.qobj()."""
        for backend in BasicAer.backends():
            with self.subTest(backend=backend):
                qobj = assemble(self.qc1)
                job = backend.run(qobj)
                self.assertEqual(job.qobj(), qobj)
