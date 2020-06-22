# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

import json
import os
from contextlib import contextmanager

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer


RESULTDIR = os.path.dirname(os.path.abspath(__file__))


def save_data(image_filename, testname):
    datafilename = 'result_test.json'
    if os.path.exists(datafilename):
        with open(datafilename, 'r') as datafile:
            data = json.load(datafile)
    else:
        data = {}
    data[image_filename] = testname
    with open(datafilename, 'w') as datafile:
        json.dump(data, datafile)


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def save_data_wrap(func, testname):
    def wrapper(*args, **kwargs):
        image_filename = kwargs['filename']
        with cwd(RESULTDIR):
            results = func(*args, **kwargs)
            save_data(image_filename, testname)
        return results

    return wrapper


class TestMatplotlibDrawer(QiskitTestCase):
    """Circuit MPL visualization"""

    def setUp(self):
        self.circuit_drawer = save_data_wrap(circuit_drawer, str(self))

    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """

        # add a register with a very long name
        qr = QuantumRegister(4, 'veryLongQuantumRegisterName')
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, 'q0')
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        self.circuit_drawer(circuit, output='mpl', filename='long_name.png')

    def test_simple_cregbundle(self):
        """Test cregbundle=True in a simple example
        See: https://github.com/Qiskit/qiskit-terra/issues/4290
        """

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.measure(qr, cr)

        self.circuit_drawer(circuit, output='mpl', filename='simple_cregbundle.png',
                            cregbundle=True)


if __name__ == '__main__':
    unittest.main(verbosity=1)
