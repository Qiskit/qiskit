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
from unittest.mock import patch


from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import visualization
import json
import os


class TestMatplotlibDrawer(QiskitTestCase):
    """Circuit MPL visualization"""

    def save_data(self, testname, filename):
        datafilename = 'result_test.json'

        if os.path.exists(datafilename):
            with open(datafilename, 'r') as datafile:
                data = json.load(datafile)
        else:
            data = {}

        data[filename] = testname

        with open(datafilename, 'w') as datafile:
            json.dump(data, datafile)

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

        with patch('qiskit.visualization.circuit_drawer') as ctx:
            visualization.circuit_drawer(circuit, output='mpl', filename='long_name.png')
        self.save_data(str(self), ctx.call_args[1]['filename'])

if __name__ == '__main__':
    unittest.main(verbosity=1)
