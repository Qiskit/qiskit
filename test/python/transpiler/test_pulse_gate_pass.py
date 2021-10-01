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

"""Transpiler pulse gate pass testing."""

from qiskit import pulse, circuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAthens


class TestPulseGate(QiskitTestCase):
    """Integration test of pulse gate pass with custom backend."""

    def setUp(self):
        super().setUp()

        with pulse.build(name="sx_q0") as custom_sx_q0:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))

        self.custom_sx_q0 = custom_sx_q0

        with pulse.build(name="sx_q1") as custom_sx_q1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(1))

        self.custom_sx_q1 = custom_sx_q1

        self.sched_param = circuit.Parameter("P0")

        with pulse.build(name="my_gate_q0") as my_gate_q0:
            pulse.shift_phase(self.sched_param, pulse.DriveChannel(0))
            pulse.play(pulse.Constant(120, 0.1), pulse.DriveChannel(0))

        self.my_gate_q0 = my_gate_q0

        with pulse.build(name="my_gate_q1") as my_gate_q1:
            pulse.shift_phase(self.sched_param, pulse.DriveChannel(1))
            pulse.play(pulse.Constant(120, 0.2), pulse.DriveChannel(1))

        self.my_gate_q1 = my_gate_q1

    def test_transpile_with_bare_backend(self):
        """Test transpile without custom calibrations."""
        backend = FakeAthens()

        qc = circuit.QuantumCircuit(2)
        qc.sx(0)
        qc.x(0)
        qc.rz(0, 0)
        qc.sx(1)
        qc.measure_all()

        transpiled_qc = transpile(qc, backend, initial_layout=[0, 1])

        ref_calibration = {}
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_custom_basis_gate(self):
        """Test transpile with custom calibrations."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add("sx", (0,), self.custom_sx_q0)
        backend.defaults().instruction_schedule_map.add("sx", (1,), self.custom_sx_q1)

        qc = circuit.QuantumCircuit(2)
        qc.sx(0)
        qc.x(0)
        qc.rz(0, 0)
        qc.sx(1)
        qc.measure_all()

        transpiled_qc = transpile(qc, backend, initial_layout=[0, 1])

        ref_calibration = {
            "sx": {
                ((0,), ()): self.custom_sx_q0,
                ((1,), ()): self.custom_sx_q1,
            }
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_instmap(self):
        """Test providing instruction schedule map."""
        instmap = FakeAthens().defaults().instruction_schedule_map
        instmap.add("sx", (0,), self.custom_sx_q0)
        instmap.add("sx", (1,), self.custom_sx_q1)

        # Inst map is renewed
        backend = FakeAthens()

        qc = circuit.QuantumCircuit(2)
        qc.sx(0)
        qc.x(0)
        qc.rz(0, 0)
        qc.sx(1)
        qc.measure_all()

        transpiled_qc = transpile(qc, backend, inst_map=instmap, initial_layout=[0, 1])

        ref_calibration = {
            "sx": {
                ((0,), ()): self.custom_sx_q0,
                ((1,), ()): self.custom_sx_q1,
            }
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_custom_gate(self):
        """Test providing non-basis gate."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add(
            "my_gate", (0,), self.my_gate_q0, arguments=["P0"]
        )
        backend.defaults().instruction_schedule_map.add(
            "my_gate", (1,), self.my_gate_q1, arguments=["P0"]
        )

        qc = circuit.QuantumCircuit(2)
        qc.append(circuit.Gate("my_gate", 1, [1.0]), [0])
        qc.append(circuit.Gate("my_gate", 1, [2.0]), [1])

        transpiled_qc = transpile(qc, backend, basis_gates=["my_gate"], initial_layout=[0, 1])

        my_gate_q0_1_0 = self.my_gate_q0.assign_parameters({self.sched_param: 1.0}, inplace=False)
        my_gate_q1_2_0 = self.my_gate_q1.assign_parameters({self.sched_param: 2.0}, inplace=False)

        ref_calibration = {
            "my_gate": {
                ((0,), (1.0,)): my_gate_q0_1_0,
                ((1,), (2.0,)): my_gate_q1_2_0,
            }
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_multiple_circuits(self):
        """Test transpile with multiple circuits with custom gate."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add(
            "my_gate", (0,), self.my_gate_q0, arguments=["P0"]
        )

        params = [0.0, 1.0, 2.0, 3.0]
        circs = []
        for param in params:
            qc = circuit.QuantumCircuit(1)
            qc.append(circuit.Gate("my_gate", 1, [param]), [0])
            circs.append(qc)

        transpiled_qcs = transpile(circs, backend, basis_gates=["my_gate"], initial_layout=[0])

        for param, transpiled_qc in zip(params, transpiled_qcs):
            my_gate_q0_x = self.my_gate_q0.assign_parameters(
                {self.sched_param: param}, inplace=False
            )
            ref_calibration = {"my_gate": {((0,), (param,)): my_gate_q0_x}}
            self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_multiple_instructions_with_different_parameters(self):
        """Test adding many instruction with different parameter binding."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add(
            "my_gate", (0,), self.my_gate_q0, arguments=["P0"]
        )

        qc = circuit.QuantumCircuit(1)
        qc.append(circuit.Gate("my_gate", 1, [1.0]), [0])
        qc.append(circuit.Gate("my_gate", 1, [2.0]), [0])
        qc.append(circuit.Gate("my_gate", 1, [3.0]), [0])

        transpiled_qc = transpile(qc, backend, basis_gates=["my_gate"], initial_layout=[0])

        my_gate_q0_1_0 = self.my_gate_q0.assign_parameters({self.sched_param: 1.0}, inplace=False)
        my_gate_q0_2_0 = self.my_gate_q0.assign_parameters({self.sched_param: 2.0}, inplace=False)
        my_gate_q0_3_0 = self.my_gate_q0.assign_parameters({self.sched_param: 3.0}, inplace=False)

        ref_calibration = {
            "my_gate": {
                ((0,), (1.0,)): my_gate_q0_1_0,
                ((0,), (2.0,)): my_gate_q0_2_0,
                ((0,), (3.0,)): my_gate_q0_3_0,
            }
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_different_qubit(self):
        """Test transpile with qubit without custom gate."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add("sx", (0,), self.custom_sx_q0)

        qc = circuit.QuantumCircuit(1)
        qc.sx(0)
        qc.measure_all()

        transpiled_qc = transpile(qc, backend, initial_layout=[3])

        self.assertDictEqual(transpiled_qc.calibrations, {})
