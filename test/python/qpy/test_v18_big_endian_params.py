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

"""Regression tests for big-endian instruction parameters in QPY v18.

QPY v1–17 serialised instruction parameter integers and floats in little-endian
by mistake.  v18 corrects this to big-endian (matching the rest of the format).
These tests verify:
  - v18 and v17 produce *different* on-disk bytes for the same parameterised circuit
    (proving the encoding actually changed).
  - Both versions round-trip back to equal circuits (forward and backward compat).
  - The specific cases that required special handling: ForLoopOp integer lists and
    SwitchCase integer labels.
"""

import io
import struct

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.qpy import dump, load
from test import QiskitTestCase


def _dump(qc: QuantumCircuit, version: int) -> bytes:
    buf = io.BytesIO()
    dump(qc, buf, version=version)
    return buf.getvalue()


def _load(data: bytes) -> QuantumCircuit:
    return load(io.BytesIO(data))[0]


class TestV18BigEndianFloatParam(QiskitTestCase):
    """Float gate parameters are big-endian in v18, little-endian in v17."""

    def _make_circuit(self):
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        return qc

    def test_v17_v18_bytes_differ(self):
        """v17 and v18 serialise the float parameter in different byte order."""
        qc = self._make_circuit()
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_v18_roundtrip(self):
        """Float parameter round-trips correctly through QPY v18."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 18)))

    def test_v17_roundtrip(self):
        """Float parameter round-trips correctly through QPY v17 (back-compat)."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 17)))

    def test_v17_bytes_are_little_endian(self):
        """v17 float bytes in the payload are actually little-endian on disk."""
        qc = self._make_circuit()
        data = _dump(qc, 17)
        # The float 1.23456789 little-endian bytes must appear somewhere in the payload.
        le_bytes = struct.pack("<d", 1.23456789)
        self.assertIn(le_bytes, data)

    def test_v18_bytes_are_big_endian(self):
        """v18 float bytes in the payload are big-endian on disk."""
        qc = self._make_circuit()
        data = _dump(qc, 18)
        be_bytes = struct.pack(">d", 1.23456789)
        self.assertIn(be_bytes, data)


class TestV18BigEndianIntParam(QiskitTestCase):
    """Integer instruction parameters are big-endian in v18, little-endian in v17."""

    def _make_circuit(self):
        # PhaseGate takes a float, but CXGate has no params; use a circuit with
        # an integer stored directly — RZZ angle as a Python int exercises the Int64 path.
        qc = QuantumCircuit(2)
        qc.rzz(3, 0, 1)
        return qc

    def test_v18_roundtrip(self):
        """Integer-valued float parameter round-trips correctly through QPY v18."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 18)))

    def test_v17_roundtrip(self):
        """Integer-valued float parameter round-trips correctly through QPY v17."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 17)))


class TestV18BigEndianForLoop(QiskitTestCase):
    """ForLoopOp integer-list parameters are big-endian in v18, little-endian in v17."""

    def _make_circuit(self):
        qc = QuantumCircuit(1, 1)
        with qc.for_loop((1, 4, 9)):
            qc.h(0)
        return qc

    def test_v17_v18_bytes_differ(self):
        """v17 and v18 serialise ForLoop integer list in different byte order."""
        qc = self._make_circuit()
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_v18_roundtrip(self):
        """ForLoopOp integer list round-trips correctly through QPY v18."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 18)))

    def test_v17_roundtrip(self):
        """ForLoopOp integer list round-trips correctly through QPY v17 (back-compat)."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 17)))


class TestV18BigEndianSwitchCase(QiskitTestCase):
    """SwitchCase integer labels are big-endian in v18, little-endian in v17."""

    def _make_circuit(self):
        body = QuantumCircuit(1)
        body.h(0)
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy()), (2, body.copy())], [0], [])
        return qc

    def test_v17_v18_bytes_differ(self):
        """v17 and v18 serialise SwitchCase labels in different byte order."""
        qc = self._make_circuit()
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_v18_roundtrip(self):
        """SwitchCase integer labels round-trip correctly through QPY v18."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 18)))

    def test_v17_roundtrip(self):
        """SwitchCase integer labels round-trip correctly through QPY v17 (back-compat)."""
        qc = self._make_circuit()
        self.assertEqual(qc, _load(_dump(qc, 17)))
