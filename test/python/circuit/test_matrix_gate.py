# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test matrix gates"""

import os
import tempfile
import unittest
import numpy

import qiskit.extensions.simulator
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.circuit import Gate
from qiskit.test import QiskitTestCase
from qiskit.transpiler import transpile, PassManager
from qiskit.transpiler.passes import BasicSwap, CXCancellation, Optimize1qGates
from qiskit import BasicAer


class TestMatrixGate(QiskitTestCase):
    """Matrix gate tests."""

    def test_init(self):
        backend = BasicAer.get_backend('qasm_simulator')
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.x(qr[0])
        qc.umatrix(matrix, qr[0])
        pm = PassManager()
        pm.append(CXCancellation())
        qc2 = transpile(qc, backend, pass_manager=pm)
        print(qc2.qasm())
        print(qc2)
        import pdb;pdb.set_trace()
        
