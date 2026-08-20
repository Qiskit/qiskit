# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPY v18 format changes compared to v17."""

import io
import struct

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.qpy import dump, load
from qiskit.qpy import formats
from qiskit.qpy.exceptions import QpyError
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


class TestV18RegisterParam(QiskitTestCase):
    """The `Register` payload gained a tag byte in v18.

    It identifies either a whole :class:`.ClassicalRegister` or a single :class:`.Clbit`.  Up to v17
    the two shared one untyped string -- a register was its bare name, a clbit was a null byte
    followed by its index in ASCII digits -- so the payload could not be identified from its own
    bytes.
    """

    #: The condition payload follows the instruction's class name, as these circuits set no label.
    GATE_NAME = b"IfElseOp"

    @staticmethod
    def _clbit_condition(index=1, num_clbits=2):
        """A circuit whose only instruction is conditioned on a single clbit."""
        qreg, creg = QuantumRegister(1, "q"), ClassicalRegister(num_clbits, "creg")
        circuit = QuantumCircuit(qreg, creg)
        body = QuantumCircuit(1)
        body.x(0)
        circuit.if_test((creg[index], True), body, [qreg[0]], [])
        return circuit

    @staticmethod
    def _register_condition(name="creg"):
        """A circuit whose only instruction is conditioned on a whole classical register."""
        qreg, creg = QuantumRegister(1, "q"), ClassicalRegister(2, name)
        circuit = QuantumCircuit(qreg, creg)
        body = QuantumCircuit(1)
        body.x(0)
        circuit.if_test((creg, 1), body, [qreg[0]], [])
        return circuit

    def _condition_payload(self, data, length):
        """The ``length`` bytes of condition payload that follow the instruction's class name."""
        start = data.rindex(self.GATE_NAME) + len(self.GATE_NAME)
        return bytes(data[start : start + length])

    def test_clbit_condition_is_tagged_in_v18(self):
        """A clbit is a tag byte plus a uint32 index from v18; ASCII digits up to v17."""
        for index in (0, 1, 3):
            with self.subTest(index=index):
                circuit = self._clbit_condition(index=index, num_clbits=4)
                legacy = b"\x00" + str(index).encode("utf8")
                self.assertEqual(self._condition_payload(_dump(circuit, 17), len(legacy)), legacy)
                tagged = struct.pack(
                    formats.REGISTER_PARAM_CLBIT_PACK, formats.REGISTER_PARAM_TAG_CLBIT, index
                )
                self.assertEqual(self._condition_payload(_dump(circuit, 18), len(tagged)), tagged)

    def test_register_condition_is_tagged_in_v18(self):
        """A register is a tag byte plus its name from v18; the bare name up to v17."""
        circuit = self._register_condition()
        self.assertEqual(self._condition_payload(_dump(circuit, 17), 4), b"creg")
        tagged = (
            struct.pack(formats.REGISTER_PARAM_TAG_PACK, formats.REGISTER_PARAM_TAG_REGISTER)
            + b"creg"
        )
        self.assertEqual(self._condition_payload(_dump(circuit, 18), len(tagged)), tagged)

    def test_null_prefixed_register_name_needs_v18(self):
        """A register name starting with a null byte is only representable from v18.

        Qiskit accepts such a name, but up to v17 it collides with the marker for a single-bit
        condition, so the reader takes the name for a bit index and fails.  The tag removes the
        ambiguity.
        """
        circuit = self._register_condition(name="\x00weird")
        with self.assertRaises((QpyError, ValueError)):
            load(io.BytesIO(_dump(circuit, 17)))
        self.assertEqual(load(io.BytesIO(_dump(circuit, 18)))[0], circuit)

    def test_v18_unknown_tag_is_rejected(self):
        """An unrecognised tag byte must fail rather than be guessed at."""
        data = bytearray(_dump(self._clbit_condition(), 18))
        data[data.rindex(self.GATE_NAME) + len(self.GATE_NAME)] = 7
        with self.assertRaises(QpyError):
            load(io.BytesIO(bytes(data)))

    def test_v18_unknown_register_name_is_rejected(self):
        """A register name that is not in the circuit must fail."""
        data = bytearray(_dump(self._register_condition(), 18))
        start = data.rindex(self.GATE_NAME) + len(self.GATE_NAME)
        data[start + 1 : start + 5] = b"zzzz"  # same length, so no size field to fix up
        with self.assertRaises(QpyError):
            load(io.BytesIO(bytes(data)))
