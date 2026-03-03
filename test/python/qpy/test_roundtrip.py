# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for python write/rust read flow and vice versa"""

import io
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.qpy.binary_io import write_circuit, read_circuit
from test import QiskitTestCase


class TestQPYRoundtrip(QiskitTestCase):
    """QPY circuit testing platform."""

    def assert_roundtrip_equal(
        self,
        circuit,
        version=None,
        annotation_factories=None,
        write_with="Python",
        read_with="Rust",
    ):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        use_rust_for_write = write_with == "Rust"
        use_rust_for_read = read_with == "Rust"
        write_circuit(
            qpy_file,
            circuit,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust_for_write,
        )
        qpy_file.seek(0)
        new_circuit = read_circuit(
            qpy_file,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust_for_read,
        )
        self.assertEqual(circuit, new_circuit)
        self.assertEqual(circuit.layout, new_circuit.layout)

    def test_simple(self):
        """Basic roundtrip test"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        self.assert_roundtrip_equal(qc, version=17)

    def test_ifelse(self):
        """Check the IfElse control flow gate passes roundtrip"""
        qc = QuantumCircuit(1, 1)
        condition = (qc.cregs[0], 0)
        body = QuantumCircuit([qc.qubits[0]])
        body.x(0)
        qc.if_else(condition, body, None, [qc.qubits[0]], [])
        self.assert_roundtrip_equal(qc, version=17)

    def test_box(self):
        """Check the BoxOp control flow gate passes roundtrip"""
        qc = QuantumCircuit(2)
        with qc.box(duration=13):
            qc.cx(0, 1)
        self.assert_roundtrip_equal(qc, version=17)

    def test_forloop(self):
        """Check the ForLoop control flow gate passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=17)

    def test_switch(self):
        """Check the SwitchOp control flow gate passes roundtrip"""
        body = QuantumCircuit(1)
        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy())], [0], [])
        qc.switch(expr.logic_not(qc.clbits[0]), [(False, body.copy())], [0], [])
        self.assert_roundtrip_equal(qc, version=17)
