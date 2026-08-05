# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPY v18 format changes compared to v17."""

import io
import struct

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.qpy import dump
from qiskit.qpy import formats
from test import QiskitTestCase


def _dump(qc: QuantumCircuit, version: int) -> bytes:
    buf = io.BytesIO()
    dump(qc, buf, version=version)
    return buf.getvalue()



class TestV17VsV18(QiskitTestCase):
    """Verify the binary-level differences between QPY v17 and v18."""

    def test_v18_smaller_than_v17_by_calibration_header(self):
        """v18 output is exactly 2 bytes smaller than v17 (CalibrationsPack removed)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        size17 = len(_dump(qc, 17))
        size18 = len(_dump(qc, 18))
        cal_header_size = struct.calcsize(formats.CALIBRATION_PACK)  # 2 bytes

        self.assertEqual(
            size17 - size18,
            cal_header_size,
            f"Expected v18 to be {cal_header_size} bytes smaller than v17, "
            f"got v17={size17} v18={size18} diff={size17 - size18}",
        )

    def test_float_param_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise float parameters in different byte order."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_float_param_v17_is_little_endian(self):
        """v17 float parameter bytes are little-endian on disk."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertIn(struct.pack("<d", 1.23456789), _dump(qc, 17))

    def test_float_param_v18_is_big_endian(self):
        """v18 float parameter bytes are big-endian on disk."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertIn(struct.pack(">d", 1.23456789), _dump(qc, 18))

    def test_for_loop_integers_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise ForLoop integer lists in different byte order."""
        qc = QuantumCircuit(1, 1)
        with qc.for_loop((1, 4, 9)):
            qc.h(0)
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_switch_case_labels_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise SwitchCase integer labels in different byte order."""
        body = QuantumCircuit(1)
        body.h(0)
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy()), (2, body.copy())], [0], [])
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))
