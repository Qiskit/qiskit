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

import ddt

from qiskit import pulse, circuit, transpile
from qiskit.providers.fake_provider import FakeAthens, FakeAthensV2
from qiskit.quantum_info.random import random_unitary
from qiskit.test import QiskitTestCase


@ddt.ddt
class TestPulseGate(QiskitTestCase):
    """Integration test of pulse gate pass with custom backend."""

    def setUp(self):
        super().setUp()

        self.sched_param = circuit.Parameter("P0")

        with pulse.build(name="sx_q0") as custom_sx_q0:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))
        self.custom_sx_q0 = custom_sx_q0

        with pulse.build(name="sx_q1") as custom_sx_q1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(1))
        self.custom_sx_q1 = custom_sx_q1

        with pulse.build(name="cx_q01") as custom_cx_q01:
            pulse.play(pulse.Constant(100, 0.4), pulse.ControlChannel(0))
        self.custom_cx_q01 = custom_cx_q01

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

    def test_transpile_with_backend_target(self):
        """Test transpile without custom calibrations from target."""
        backend = FakeAthensV2()
        target = backend.target

        qc = circuit.QuantumCircuit(2)
        qc.sx(0)
        qc.x(0)
        qc.rz(0, 0)
        qc.sx(1)
        qc.measure_all()

        transpiled_qc = transpile(qc, initial_layout=[0, 1], target=target)

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

    def test_transpile_with_custom_basis_gate_in_target(self):
        """Test transpile with custom calibrations."""
        backend = FakeAthensV2()
        target = backend.target
        target["sx"][(0,)].calibration = self.custom_sx_q0
        target["sx"][(1,)].calibration = self.custom_sx_q1

        qc = circuit.QuantumCircuit(2)
        qc.sx(0)
        qc.x(0)
        qc.rz(0, 0)
        qc.sx(1)
        qc.measure_all()

        transpiled_qc = transpile(qc, initial_layout=[0, 1], target=target)

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

    def test_transpile_with_parameterized_custom_gate(self):
        """Test providing non-basis gate, which is kept parameterized throughout transpile."""
        backend = FakeAthens()
        backend.defaults().instruction_schedule_map.add(
            "my_gate", (0,), self.my_gate_q0, arguments=["P0"]
        )

        param = circuit.Parameter("new_P0")
        qc = circuit.QuantumCircuit(1)
        qc.append(circuit.Gate("my_gate", 1, [param]), [0])

        transpiled_qc = transpile(qc, backend, basis_gates=["my_gate"], initial_layout=[0])

        my_gate_q0_p = self.my_gate_q0.assign_parameters({self.sched_param: param}, inplace=False)

        ref_calibration = {
            "my_gate": {
                ((0,), (param,)): my_gate_q0_p,
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

    @ddt.data(0, 1, 2, 3)
    def test_transpile_with_both_instmap_and_empty_target(self, opt_level):
        """Test when instmap and target are both provided
        and only instmap contains custom schedules.

        Test case from Qiskit/qiskit-terra/#9489
        """
        instmap = FakeAthens().defaults().instruction_schedule_map
        instmap.add("sx", (0,), self.custom_sx_q0)
        instmap.add("sx", (1,), self.custom_sx_q1)
        instmap.add("cx", (0, 1), self.custom_cx_q01)

        # This doesn't have custom schedule definition
        target = FakeAthensV2().target

        qc = circuit.QuantumCircuit(2)
        qc.append(random_unitary(4, seed=123), [0, 1])
        qc.measure_all()

        transpiled_qc = transpile(
            qc,
            optimization_level=opt_level,
            basis_gates=["sx", "rz", "x", "cx"],
            inst_map=instmap,
            target=target,
            initial_layout=[0, 1],
        )
        ref_calibration = {
            "sx": {
                ((0,), ()): self.custom_sx_q0,
                ((1,), ()): self.custom_sx_q1,
            },
            "cx": {
                ((0, 1), ()): self.custom_cx_q01,
            },
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    @ddt.data(0, 1, 2, 3)
    def test_transpile_with_instmap_with_v2backend(self, opt_level):
        """Test when instmap is provided with V2 backend.

        Test case from Qiskit/qiskit-terra/#9489
        """
        instmap = FakeAthens().defaults().instruction_schedule_map
        instmap.add("sx", (0,), self.custom_sx_q0)
        instmap.add("sx", (1,), self.custom_sx_q1)
        instmap.add("cx", (0, 1), self.custom_cx_q01)

        qc = circuit.QuantumCircuit(2)
        qc.append(random_unitary(4, seed=123), [0, 1])
        qc.measure_all()

        transpiled_qc = transpile(
            qc,
            FakeAthensV2(),
            optimization_level=opt_level,
            inst_map=instmap,
            initial_layout=[0, 1],
        )
        ref_calibration = {
            "sx": {
                ((0,), ()): self.custom_sx_q0,
                ((1,), ()): self.custom_sx_q1,
            },
            "cx": {
                ((0, 1), ()): self.custom_cx_q01,
            },
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    @ddt.data(0, 1, 2, 3)
    def test_transpile_with_instmap_with_v2backend_with_custom_gate(self, opt_level):
        """Test when instmap is provided with V2 backend.

        In this test case, instmap contains a custom gate which doesn't belong to
        Qiskit standard gate. Target must define a custom gete on the fly
        to reflect user-provided instmap.

        Test case from Qiskit/qiskit-terra/#9489
        """
        with pulse.build(name="custom") as rabi12:
            pulse.play(pulse.Constant(100, 0.4), pulse.DriveChannel(0))

        instmap = FakeAthens().defaults().instruction_schedule_map
        instmap.add("rabi12", (0,), rabi12)

        gate = circuit.Gate("rabi12", 1, [])
        qc = circuit.QuantumCircuit(1)
        qc.append(gate, [0])
        qc.measure_all()

        transpiled_qc = transpile(
            qc,
            FakeAthensV2(),
            optimization_level=opt_level,
            inst_map=instmap,
            initial_layout=[0],
        )
        ref_calibration = {
            "rabi12": {
                ((0,), ()): rabi12,
            }
        }
        self.assertDictEqual(transpiled_qc.calibrations, ref_calibration)

    def test_transpile_with_instmap_not_mutate_backend(self):
        """Do not override default backend target when transpile with inst map.

        Providing an instmap for the transpile arguments may override target,
        which might be pulled from the provided backend instance.
        This should not override the source object since the same backend may
        be used for future transpile without intention of instruction overriding.
        """
        backend = FakeAthensV2()
        original_sx0 = backend.target["sx"][(0,)].calibration

        instmap = FakeAthens().defaults().instruction_schedule_map
        instmap.add("sx", (0,), self.custom_sx_q0)

        qc = circuit.QuantumCircuit(1)
        qc.sx(0)
        qc.measure_all()

        transpiled_qc = transpile(
            qc,
            FakeAthensV2(),
            inst_map=instmap,
            initial_layout=[0],
        )
        self.assertTrue(transpiled_qc.has_calibration_for(transpiled_qc.data[0]))

        self.assertEqual(
            backend.target["sx"][(0,)].calibration,
            original_sx0,
        )
