# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for python write/rust read flow and vice versa"""

import io
from qiskit.circuit import QuantumCircuit
from qiskit.qpy import dump, load
from test import QiskitTestCase  # pylint: disable=wrong-import-order


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
        dump(
            circuit,
            qpy_file,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust_for_write,
        )
        qpy_file.seek(0)
        new_circuit = load(
            qpy_file, annotation_factories=annotation_factories, use_rust=use_rust_for_read
        )[0]

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
        """Check the IfElse conditional gate passes roundtrip"""
        qc = QuantumCircuit(1, 1)
        condition = (qc.cregs[0], 0)
        body = QuantumCircuit([qc.qubits[0]])
        body.x(0)
        qc.if_else(condition, body, None, [qc.qubits[0]], [])
        self.assert_roundtrip_equal(qc, version=17)
