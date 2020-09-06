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

# pylint: disable=missing-docstring

"""Non-string identifiers for circuit and record identifiers test"""

import unittest

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import assemble
from qiskit.test import QiskitTestCase


class TestQobjIdentifiers(QiskitTestCase):
    """Check the Qobj compiled for different backends create names properly"""

    def setUp(self):
        qr = QuantumRegister(2, name="qr2")
        cr = ClassicalRegister(2, name=None)
        qc = QuantumCircuit(qr, cr, name="qc10")
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qr_name = qr.name
        self.cr_name = cr.name
        self.circuits = [qc]

    def test_builtin_qasm_simulator_py(self):
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))

    def test_builtin_qasm_simulator(self):
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))

    def test_builtin_unitary_simulator_py(self):
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))


if __name__ == '__main__':
    unittest.main(verbosity=2)
