# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the schedule block qpy loading and saving."""

import io
import struct

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.providers.fake_provider import FakeHanoi, FakeSherbrooke
from qiskit.exceptions import QiskitError
from qiskit.qpy import dump, load, formats
from qiskit.qpy.common import QPY_VERSION
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager, TranspileLayout
from qiskit.transpiler import passes
from qiskit.compiler import transpile


class QpyCircuitTestCase(QiskitTestCase):
    """QPY schedule testing platform."""

    def assert_layout_equal(self, first_qc, first_layout, second_qc, second_layout):
        """Assert layout equality up to `BitLocations`
        Args:
            first_qc (QuantumCircuit): a quantum circuit
            first_layout (QuantumCircuit): a quantum circuit layout
            second_qc (QuantumCircuit): other quantum circuit
            second_layout (QuantumCircuit): other quantum circuit layout

        Returns:
            bool: `self` and `other` are equal.
        """
        if first_layout is None and second_layout is None:
            self.assertTrue(first_layout == second_layout)
        else:
            self.assertEqual(first_layout._p2v.keys(), second_layout._p2v.keys())
            for k in first_layout._p2v:
                self.assertEqual(
                    first_qc.find_bit(first_layout._p2v[k]),
                    second_qc.find_bit(second_layout._p2v[k]),
                )

    def assert_roundtrip_equal(self, circuit):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        dump(circuit, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]

        self.assertEqual(circuit, new_circuit)
        if circuit is None or new_circuit is None:
            self.assertEqual(circuit, new_circuit)
        elif circuit.layout is None or new_circuit.layout is None:
            self.assertEqual(circuit.layout, new_circuit.layout)
        else:
            self.assert_layout_equal(
                circuit, circuit.layout.final_layout, new_circuit, new_circuit.layout.final_layout
            )


@ddt
class TestCalibrationPasses(QpyCircuitTestCase):
    """QPY round-trip test case of transpiled circuits with pulse level optimization."""

    def setUp(self):
        super().setUp()
        # This backend provides CX(0,1) with native ECR direction.
        self.inst_map = FakeHanoi().defaults().instruction_schedule_map

    @data(0.1, 0.7, 1.5)
    def test_rzx_calibration(self, angle):
        """RZX builder calibration pass with echo."""
        pass_ = passes.RZXCalibrationBuilder(self.inst_map)
        pass_manager = PassManager(pass_)
        test_qc = QuantumCircuit(2)
        test_qc.rzx(angle, 0, 1)
        rzx_qc = pass_manager.run(test_qc)

        self.assert_roundtrip_equal(rzx_qc)

    @data(0.1, 0.7, 1.5)
    def test_rzx_calibration_echo(self, angle):
        """RZX builder calibration pass without echo."""
        pass_ = passes.RZXCalibrationBuilderNoEcho(self.inst_map)
        pass_manager = PassManager(pass_)
        test_qc = QuantumCircuit(2)
        test_qc.rzx(angle, 0, 1)
        rzx_qc = pass_manager.run(test_qc)

        self.assert_roundtrip_equal(rzx_qc)


class TestVersions(QpyCircuitTestCase):
    """Test version handling in qpy."""

    def test_invalid_qpy_version(self):
        """Test a descriptive exception is raised if QPY version is too new."""
        with io.BytesIO() as buf:
            buf.write(
                struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", QPY_VERSION + 4, 42, 42, 1, 2)
            )
            buf.seek(0)
            with self.assertRaisesRegex(QiskitError, str(QPY_VERSION + 4)):
                load(buf)


@ddt
class TestLayout(QpyCircuitTestCase):
    """Test circuit serialization for layout preservation."""

    @data(0, 1, 2, 3)
    def test_transpile_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = FakeSherbrooke()
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_with_routing(self, opt_level):
        """Test full layout with routing is preserved."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = FakeSherbrooke()
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_layout_explicit_None_final_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = FakeSherbrooke()
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.layout.final_layout = None
        self.assert_roundtrip_equal(tqc)

    def test_empty_layout(self):
        """Test an empty layout is preserved correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc._layout = TranspileLayout(None, None, None)
        self.assert_roundtrip_equal(qc)

    @data(0, 1, 2, 3)
    def test_custom_register_name(self, opt_level):
        """Test layout preserved with custom register names."""
        qr = QuantumRegister(5, name="abc123")
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = FakeSherbrooke()
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_no_register(self, opt_level):
        """Test layout preserved with no register."""
        qubits = [Qubit(), Qubit()]
        qc = QuantumCircuit(qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = FakeSherbrooke()
        tqc = transpile(qc, backend, optimization_level=opt_level)
        # Manually validate to deal with qubit equality needing exact objects
        qpy_file = io.BytesIO()
        dump(tqc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(tqc, new_circuit)
        initial_layout_old = tqc.layout.initial_layout.get_physical_bits()
        initial_layout_new = new_circuit.layout.initial_layout.get_physical_bits()
        for i in initial_layout_old:
            self.assertIsInstance(initial_layout_old[i], Qubit)
            self.assertIsInstance(initial_layout_new[i], Qubit)
            if initial_layout_old[i] in qc.qubits and initial_layout_new[i] in qc.qubits:
                self.assertEqual(initial_layout_new[i], initial_layout_old[i])

        self.assertEqual(
            list(tqc.layout.input_qubit_mapping.values()),
            list(new_circuit.layout.input_qubit_mapping.values()),
        )
        self.assertEqual(tqc.layout.final_layout, new_circuit.layout.final_layout)
