# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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
"""Tests for visualization tools."""

import os
import unittest
import numpy

from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.tools.visualization import latex_drawer
from qiskit.tools.visualization import circuit_drawer
from .common import QiskitTestCase


class TestLatexDrawer(QiskitTestCase):
    """QISKit latex drawer tests."""

    def setUp(self):
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit('latex_test', [qr], [cr])
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[1], cr[1])
        qc.x(qr[1]).c_if(cr, 1)
        qc.measure(qr, cr)
        self.qp = qp
        self.qc = qc
        self.qobj = qp.compile(['latex_test'])

    def test_latex_drawer(self):
        filename = self._get_resource_path('test_latex_drawer.tex')
        try:
            latex_drawer(self.qc, filename)
        except Exception:
            if os.path.exists(filename):
                os.remove(filename)
            raise

    def test_teleport(self):
        filename = self._get_resource_path('test_teleport.tex')
        QPS_SPECS = {
            "circuits": [{
                "name": "teleport",
                "quantum_registers": [{
                    "name": "q",
                    "size": 3
                }],
                "classical_registers": [
                    {"name": "c0",
                     "size": 1},
                    {"name": "c1",
                     "size": 1},
                    {"name": "c2",
                     "size": 1},
                ]}]
        }

        qp = QuantumProgram(specs=QPS_SPECS)
        qc = qp.get_circuit("teleport")
        q = qp.get_quantum_register("q")
        c0 = qp.get_classical_register("c0")
        c1 = qp.get_classical_register("c1")
        c2 = qp.get_classical_register("c2")

        # Prepare an initial state
        qc.u3(0.3, 0.2, 0.1, q[0])

        # Prepare a Bell pair
        qc.h(q[1])
        qc.cx(q[1], q[2])

        # Barrier following state preparation
        qc.barrier(q)

        # Measure in the Bell basis
        qc.cx(q[0], q[1])
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        qc.measure(q[1], c1[0])

        # Apply a correction
        qc.z(q[2]).c_if(c0, 1)
        qc.x(q[2]).c_if(c1, 1)
        qc.measure(q[2], c2[0])
        try:
            latex_drawer(qc, filename)
        except Exception:
            if os.path.exists(filename):
                os.remove(filename)
            raise


class TestCircuitDrawer(QiskitTestCase):
    """QISKit circuit drawer tests."""

    def setUp(self):
        qr = QuantumRegister('qr', 2)
        cr = ClassicalRegister('cr', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[1], cr[1])
        qc.x(qr[1]).c_if(cr, 1)
        qc.measure(qr, cr)
        self.qc = qc

    def test_teleport_image(self):
        im = circuit_drawer(self.qc)
        if im:
            pix = numpy.array(im)
            self.assertEqual(pix.shape, (260, 701, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
