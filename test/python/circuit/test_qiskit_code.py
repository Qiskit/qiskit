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

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class."""

import os
import tempfile
import unittest
import numpy

import qiskit.extensions.simulator
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase
from qiskit.extensions.unitary import UnitaryGate


class TestCircuitQiskitCode(QiskitTestCase):
    """QuantumCircuit Qiskit tests."""

    def test_circuit_qiskit_code(self):
        """Test circuit qiskit_code() method.
        """
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.u1(0.3, qr1[0])
        qc.u2(0.2, 0.1, qr2[0])
        qc.u3(0.3, 0.2, 0.1, qr2[1])
        qc.s(qr2[1])
        qc.sdg(qr2[1])
        qc.cx(qr1[0], qr2[1])
        qc.barrier(qr2)
        qc.cx(qr2[1], qr1[0])
        qc.h(qr2[1])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        qc.barrier(qr1, qr2)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        expected_python = """qr1 = QuantumRegister(1, 'qr1')\n""" + \
                          """qr2 = QuantumRegister(2, 'qr2')\n""" + \
                          """cr = ClassicalRegister(3, 'cr')\n""" + \
                          """gcn_qc = QuantumCircuit(qr1, qr2, cr)\n""" + \
                          """gcn_qc.u1(0.300000000000000, qr1[0])\n""" + \
                          """gcn_qc.u2(0.200000000000000, 0.100000000000000, qr2[0])\n""" + \
                          """gcn_qc.u3(0.300000000000000, 0.200000000000000, 0.100000000000000, qr2[1])\n""" + \
                          """gcn_qc.s(qr2[1])\n""" + \
                          """gcn_qc.sdg(qr2[1])\n""" + \
                          """gcn_qc.cx(qr1[0], qr2[1])\n""" + \
                          """gcn_qc.barrier(qr2[0], qr2[1])\n""" + \
                          """gcn_qc.cx(qr2[1], qr1[0])\n""" + \
                          """gcn_qc.h(qr2[1])\n""" + \
                          """gcn_qc.x(qr2[1]).c_if(cr, 0)\n""" + \
                          """gcn_qc.y(qr1[0]).c_if(cr, 1)\n""" + \
                          """gcn_qc.z(qr1[0]).c_if(cr, 2)\n""" + \
                          """gcn_qc.barrier(qr1[0], qr2[0], qr2[1])\n""" + \
                          """gcn_qc.measure(qr1[0], cr[0])\n""" + \
                          """gcn_qc.measure(qr2[0], cr[1])\n""" + \
                          """gcn_qc.measure(qr2[1], cr[2])\n"""
        self.assertEqual(qc.qiskit_code(), expected_python)
        qr1 = QuantumRegister(3, 'qr1')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr1, cr)
        qc.measure(qr1, cr)
        expected_python = """qr1 = QuantumRegister(3, 'qr1')\n""" + \
                          """cr = ClassicalRegister(3, 'cr')\n""" + \
                          """gcn_qc = QuantumCircuit(qr1, cr)\n""" + \
                          """gcn_qc.measure(qr1[0], cr[0])\n""" + \
                          """gcn_qc.measure(qr1[1], cr[1])\n""" + \
                          """gcn_qc.measure(qr1[2], cr[2])\n"""
        self.assertEqual(qc.qiskit_code(), expected_python)
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.append(UnitaryGate(matrix), [qr[0]])
        expected_python = """qr = QuantumRegister(3, 'qr')\n""" + \
                          """cr = ClassicalRegister(3, 'cr')\n""" + \
                          """gcn_qc = QuantumCircuit(qr, cr)\n""" + \
                          """matrix = np.array([[1.+0.j, 0.+0.j],\n""" + \
                          """       [0.+0.j, 1.+0.j]])\n""" + \
                          """gcn_qc.append(UnitaryGate(matrix), [qr[0]])\n"""
        self.assertEqual(qc.qiskit_code(), expected_python)
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr, cr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmax, sigmay)
        uni2q = UnitaryGate(matrix, label='test')
        qc.append(uni2q, [qr[0], qr[1]])
        expected_python = """qr = QuantumRegister(3, 'qr')\n""" + \
                          """cr = ClassicalRegister(3, 'cr')\n""" + \
                          """gcn_qc = QuantumCircuit(qr, cr)\n""" + \
                          """matrix = np.array([[0.+0.j, 0.-0.j, 0.+0.j, 0.-1.j],\n""" + \
                          """       [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],\n""" + \
                          """       [0.+0.j, 0.-1.j, 0.+0.j, 0.-0.j],\n""" + \
                          """       [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]])\n""" + \
                          """gcn_qc.append(UnitaryGate(matrix, label='test'), [qr[0], qr[1]])\n"""
        self.assertEqual(qc.qiskit_code(), expected_python)
        qr = QuantumRegister(3, 'qr%')
        cr = ClassicalRegister(3, 'gcn_qc')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[1])
        expected_python = """grn_qr = QuantumRegister(3, 'grn_qr')\n""" + \
                          """grn_gcn_qc = ClassicalRegister(3, 'grn_gcn_qc')\n""" + \
                          """gcn_qc = QuantumCircuit(grn_qr, grn_gcn_qc)\n""" + \
                          """gcn_qc.h(grn_qr[1])\n"""
        self.assertEqual(qc.qiskit_code(), expected_python)
