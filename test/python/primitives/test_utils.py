# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for utilities of Primitives."""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives.utils import final_measurement_mapping
from qiskit.test import QiskitTestCase


class TestMapping(QiskitTestCase):
    """Test final_measurement_mapping

    Source:
        https://github.com/Qiskit-Partners/mthree/blob/03ea16fa0f06a9b28e0a19450d88a49501e2c276/mthree/test/test_meas_mapping.py
    """

    def test_empty_circ(self):
        """Empty circuit has no mapping"""
        qc = QuantumCircuit()
        self.assertDictEqual(final_measurement_mapping(qc), {})

    def test_sime_circ(self):
        """Just measures"""
        qc = QuantumCircuit(5)
        qc.measure_all()
        self.assertDictEqual(final_measurement_mapping(qc), {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    def test_simple2_circ(self):
        """Meas followed by Hadamards"""
        qc = QuantumCircuit(5)
        qc.measure_all()
        qc.h(range(5))
        self.assertDictEqual(final_measurement_mapping(qc), {})

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
        self.assertDictEqual(final_measurement_mapping(qc), {2: 2, 3: 3, 1: 4})

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
        self.assertDictEqual(final_measurement_mapping(qc), {2: 2, 3: 3, 1: 4})

    def test_mapping_w_delays(self):
        """Check that measurements followed by delays get in the mapping"""
        qc = QuantumCircuit(2, 2)
        qc.measure(0, 1)
        qc.delay(10, 0)
        qc.measure(1, 0)
        qc.barrier()

        maps = final_measurement_mapping(qc)
        self.assertDictEqual(maps, {1: 0, 0: 1})
