# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QuantumCircuit final measurement mapping"""

import unittest
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestCircuitMeasMapping(QiskitTestCase):
    """QuantumCircuit final measurement mapping tests"""

    def test_empty_circ(self):
        """Empty circuit has no mapping"""
        qc = QuantumCircuit()
        self.assertEqual(qc.final_measurement_mapping(), {})

    def test_simple_circ(self):
        """Just measures"""
        qc = QuantumCircuit(5)
        qc.measure_all()
        self.assertEqual(qc.final_measurement_mapping(), {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    def test_simple2_circ(self):
        """Meas followed by Hadamards"""
        qc = QuantumCircuit(5)
        qc.measure_all()
        qc.h(range(5))
        self.assertEqual(qc.final_measurement_mapping(), {})

    def test_multi_qreg(self):
        """Test multiple qregs"""
        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(3, "q2")
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr1, qr2, cr)

        qc.h(range(5))
        qc.measure(0, 0)
        qc.h(range(5))
        qc.measure(range(2, 4), range(2, 4))
        qc.barrier(range(5))
        qc.measure(1, 4)
        self.assertEqual(qc.final_measurement_mapping(), {2: 2, 3: 3, 1: 4})

    def test_multi_creg(self):
        """Test multiple qregs"""
        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(3, "q2")
        cr1 = ClassicalRegister(3, "c1")
        cr2 = ClassicalRegister(2, "c2")
        qc = QuantumCircuit(qr1, qr2, cr1, cr2)

        qc.h(range(5))
        qc.measure(0, 0)
        qc.h(range(5))
        qc.measure(range(2, 4), range(2, 4))
        qc.barrier(range(5))
        qc.measure(1, 4)
        self.assertEqual(qc.final_measurement_mapping(), {2: 2, 3: 3, 1: 4})


if __name__ == "__main__":
    unittest.main()
