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

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import FakeHanoi
from qiskit.qpy import dump, load
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes


class QpyCircuitTestCase(QiskitTestCase):
    """QPY schedule testing platform."""

    def assert_roundtrip_equal(self, circuit):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        dump(circuit, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]

        self.assertEqual(circuit, new_circuit)


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
