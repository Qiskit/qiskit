# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Compiler Test."""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.qobj.qobj import Qobj
from qiskit import compile
from qiskit.tools.backends import AbstractBackend
from qiskit.test import QiskitTestCase


class TestBackend(QiskitTestCase):
    """Test the use of abstract backend"""

    def test_compile(self):
        """Test compile using AbstractBackend.

        If all correct some should exists.
        """
        backend = AbstractBackend(4,
                                  [[0, 1], [1, 0], [1, 2],
                                   [2, 1], [2, 3], [3, 2]])

        qr = QuantumRegister(4, name='q')
        cr = ClassicalRegister(4, name='c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[2])
        qc.measure(qr, cr)
        qobj = compile(qc, backend)
        self.assertIsInstance(qobj, Qobj)


if __name__ == '__main__':
    unittest.main(verbosity=2)
