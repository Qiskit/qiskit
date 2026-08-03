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

"""Regression tests for CalibrationsPack removal in QPY v18."""

import io
import struct

from qiskit.circuit import QuantumCircuit
from qiskit.qpy import dump, load
from qiskit.qpy import formats
from test import QiskitTestCase


def _make_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


class TestV18CalibrationsAbsent(QiskitTestCase):
    """
    v18 files must be exactly 2 bytes smaller than v17 files for the same circuit
    (the 2 bytes being the dropped CALIBRATION_PACK header: struct "!H" = uint16).
    """

    def test_v18_smaller_than_v17_by_calibration_header(self):
        """v18 output is exactly 2 bytes smaller than v17 (CalibrationsPack removed)."""
        qc = _make_bell()

        buf17 = io.BytesIO()
        dump(qc, buf17, version=17)
        buf18 = io.BytesIO()
        dump(qc, buf18, version=18)

        size17 = len(buf17.getvalue())
        size18 = len(buf18.getvalue())
        cal_header_size = struct.calcsize(formats.CALIBRATION_PACK)  # 2 bytes

        self.assertEqual(
            size17 - size18,
            cal_header_size,
            f"Expected v18 to be {cal_header_size} bytes smaller than v17, "
            f"got v17={size17} v18={size18} diff={size17 - size18}",
        )

    def test_v18_roundtrip(self):
        """Bell circuit round-trips correctly through QPY v18."""
        qc = _make_bell()
        buf = io.BytesIO()
        dump(qc, buf, version=18)
        buf.seek(0)
        loaded = load(buf)[0]
        self.assertEqual(qc, loaded)

    def test_v17_roundtrip(self):
        """Bell circuit round-trips correctly through QPY v17 (back-compat baseline)."""
        qc = _make_bell()
        buf = io.BytesIO()
        dump(qc, buf, version=17)
        buf.seek(0)
        loaded = load(buf)[0]
        self.assertEqual(qc, loaded)
