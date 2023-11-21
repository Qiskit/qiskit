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

# pylint: disable=missing-docstring

import math

from qiskit.circuit.library import (
    RZGate,
    SXGate,
    XGate,
    CXGate,
    RYGate,
    RXGate,
    RXXGate,
    RGate,
    IGate,
    ECRGate,
    UGate,
    CCXGate,
    RZXGate,
    CZGate,
)
from qiskit.circuit import IfElseOp, ForLoopOp, WhileLoopOp, SwitchCaseOp
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameter import Parameter
from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationPublisher, ScheduleDef
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import Target
from qiskit.transpiler import InstructionProperties
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import (
    FakeBackendV2,
    FakeMumbaiFractionalCX,
    FakeVigo,
    FakeNairobi,
    FakeGeneva,
)


class TestTarget(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.fake_backend = FakeBackendV2()
        self.fake_backend_target = self.fake_backend.target
        self.theta = Parameter("theta")
        self.phi = Parameter("phi")
        self.ibm_target = Target()
        i_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(IGate(), i_props)
        rz_props = {
            (0,): InstructionProperties(duration=0, error=0),
            (1,): InstructionProperties(duration=0, error=0),
            (2,): InstructionProperties(duration=0, error=0),
            (3,): InstructionProperties(duration=0, error=0),
            (4,): InstructionProperties(duration=0, error=0),
        }
        self.ibm_target.add_instruction(RZGate(self.theta), rz_props)
        sx_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(SXGate(), sx_props)
        x_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(XGate(), x_props)
        cx_props = {
            (3, 4): InstructionProperties(duration=270.22e-9, error=0.00713),
            (4, 3): InstructionProperties(duration=305.77e-9, error=0.00713),
            (3, 1): InstructionProperties(duration=462.22e-9, error=0.00929),
            (1, 3): InstructionProperties(duration=497.77e-9, error=0.00929),
            (1, 2): InstructionProperties(duration=227.55e-9, error=0.00659),
            (2, 1): InstructionProperties(duration=263.11e-9, error=0.00659),
            (0, 1): InstructionProperties(duration=519.11e-9, error=0.01201),
            (1, 0): InstructionProperties(duration=554.66e-9, error=0.01201),
        }
        self.ibm_target.add_instruction(CXGate(), cx_props)
        measure_props = {
            (0,): InstructionProperties(duration=5.813e-6, error=0.0751),
            (1,): InstructionProperties(duration=5.813e-6, error=0.0225),
            (2,): InstructionProperties(duration=5.813e-6, error=0.0146),
            (3,): InstructionProperties(duration=5.813e-6, error=0.0215),
            (4,): InstructionProperties(duration=5.813e-6, error=0.0333),
        }
        self.ibm_target.add_instruction(Measure(), measure_props)

        self.aqt_target = Target(description="AQT Target")
        rx_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RXGate(self.theta), rx_props)
        ry_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RYGate(self.theta), ry_props)
        rz_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RZGate(self.theta), rz_props)
        r_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RGate(self.theta, self.phi), r_props)
        rxx_props = {
            (0, 1): None,
            (0, 2): None,
            (0, 3): None,
            (0, 4): None,
            (1, 0): None,
            (2, 0): None,
            (3, 0): None,
            (4, 0): None,
            (1, 2): None,
            (1, 3): None,
            (1, 4): None,
            (2, 1): None,
            (3, 1): None,
            (4, 1): None,
            (2, 3): None,
            (2, 4): None,
            (3, 2): None,
            (4, 2): None,
            (3, 4): None,
            (4, 3): None,
        }
        self.aqt_target.add_instruction(RXXGate(self.theta), rxx_props)
        measure_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(Measure(), measure_props)
        self.empty_target = Target()
        self.ideal_sim_target = Target(num_qubits=3, description="Ideal Simulator")
        self.lam = Parameter("lam")
        for inst in [
            UGate(self.theta, self.phi, self.lam),
            RXGate(self.theta),
            RYGate(self.theta),
            RZGate(self.theta),
            CXGate(),
            ECRGate(),
            CCXGate(),
            Measure(),
        ]:
            self.ideal_sim_target.add_instruction(inst, {None: None})

    def test_qargs(self):
        self.assertEqual(set(), self.empty_target.qargs)
        expected_ibm = {
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (3, 4),
            (4, 3),
            (3, 1),
            (1, 3),
            (1, 2),
            (2, 1),
            (0, 1),
            (1, 0),
        }
        self.assertEqual(expected_ibm, self.ibm_target.qargs)
        expected_aqt = {
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (3, 1),
            (4, 1),
            (2, 3),
            (2, 4),
            (3, 2),
            (4, 2),
            (3, 4),
            (4, 3),
        }
        self.assertEqual(expected_aqt, self.aqt_target.qargs)
        expected_fake = {
            (0,),
            (1,),
            (0, 1),
            (1, 0),
        }
        self.assertEqual(expected_fake, self.fake_backend_target.qargs)
        self.assertEqual(None, self.ideal_sim_target.qargs)

    def test_qargs_for_operation_name(self):
        with self.assertRaises(KeyError):
            self.empty_target.qargs_for_operation_name("rz")
        self.assertEqual(
            self.ibm_target.qargs_for_operation_name("rz"), {(0,), (1,), (2,), (3,), (4,)}
        )
        self.assertEqual(
            self.aqt_target.qargs_for_operation_name("rz"), {(0,), (1,), (2,), (3,), (4,)}
        )
        self.assertEqual(self.fake_backend_target.qargs_for_operation_name("cx"), {(0, 1)})
        self.assertEqual(
            self.fake_backend_target.qargs_for_operation_name("ecr"),
            {
                (1, 0),
            },
        )
        self.assertEqual(self.ideal_sim_target.qargs_for_operation_name("cx"), None)

    def test_instruction_names(self):
        self.assertEqual(self.empty_target.operation_names, set())
        self.assertEqual(self.ibm_target.operation_names, {"rz", "id", "sx", "x", "cx", "measure"})
        self.assertEqual(self.aqt_target.operation_names, {"rz", "ry", "rx", "rxx", "r", "measure"})
        self.assertEqual(
            self.fake_backend_target.operation_names, {"u", "cx", "measure", "ecr", "rx_30", "rx"}
        )
        self.assertEqual(
            self.ideal_sim_target.operation_names,
            {"u", "rz", "ry", "rx", "cx", "ecr", "ccx", "measure"},
        )

    def test_operations(self):
        self.assertEqual(self.empty_target.operations, [])
        ibm_expected = [RZGate(self.theta), IGate(), SXGate(), XGate(), CXGate(), Measure()]
        for gate in ibm_expected:
            self.assertIn(gate, self.ibm_target.operations)
        aqt_expected = [
            RZGate(self.theta),
            RXGate(self.theta),
            RYGate(self.theta),
            RGate(self.theta, self.phi),
            RXXGate(self.theta),
        ]
        for gate in aqt_expected:
            self.assertIn(gate, self.aqt_target.operations)
        fake_expected = [
            UGate(self.fake_backend._theta, self.fake_backend._phi, self.fake_backend._lam),
            CXGate(),
            Measure(),
            ECRGate(),
            RXGate(math.pi / 6),
            RXGate(self.fake_backend._theta),
        ]
        for gate in fake_expected:
            self.assertIn(gate, self.fake_backend_target.operations)
        ideal_sim_expected = [
            UGate(self.theta, self.phi, self.lam),
            RXGate(self.theta),
            RYGate(self.theta),
            RZGate(self.theta),
            CXGate(),
            ECRGate(),
            CCXGate(),
            Measure(),
        ]
        for gate in ideal_sim_expected:
            self.assertIn(gate, self.ideal_sim_target.operations)

    def test_instructions(self):
        self.assertEqual(self.empty_target.instructions, [])
        ibm_expected = [
            (IGate(), (0,)),
            (IGate(), (1,)),
            (IGate(), (2,)),
            (IGate(), (3,)),
            (IGate(), (4,)),
            (RZGate(self.theta), (0,)),
            (RZGate(self.theta), (1,)),
            (RZGate(self.theta), (2,)),
            (RZGate(self.theta), (3,)),
            (RZGate(self.theta), (4,)),
            (SXGate(), (0,)),
            (SXGate(), (1,)),
            (SXGate(), (2,)),
            (SXGate(), (3,)),
            (SXGate(), (4,)),
            (XGate(), (0,)),
            (XGate(), (1,)),
            (XGate(), (2,)),
            (XGate(), (3,)),
            (XGate(), (4,)),
            (CXGate(), (3, 4)),
            (CXGate(), (4, 3)),
            (CXGate(), (3, 1)),
            (CXGate(), (1, 3)),
            (CXGate(), (1, 2)),
            (CXGate(), (2, 1)),
            (CXGate(), (0, 1)),
            (CXGate(), (1, 0)),
            (Measure(), (0,)),
            (Measure(), (1,)),
            (Measure(), (2,)),
            (Measure(), (3,)),
            (Measure(), (4,)),
        ]
        self.assertEqual(ibm_expected, self.ibm_target.instructions)
        ideal_sim_expected = [
            (UGate(self.theta, self.phi, self.lam), None),
            (RXGate(self.theta), None),
            (RYGate(self.theta), None),
            (RZGate(self.theta), None),
            (CXGate(), None),
            (ECRGate(), None),
            (CCXGate(), None),
            (Measure(), None),
        ]
        self.assertEqual(ideal_sim_expected, self.ideal_sim_target.instructions)

    def test_instruction_properties(self):
        i_gate_2 = self.ibm_target.instruction_properties(2)
        self.assertEqual(i_gate_2.error, 0.0004003)
        self.assertIsNone(self.ideal_sim_target.instruction_properties(4))

    def test_get_instruction_from_name(self):
        with self.assertRaises(KeyError):
            self.empty_target.operation_from_name("measure")
        self.assertEqual(self.ibm_target.operation_from_name("measure"), Measure())
        self.assertEqual(self.fake_backend_target.operation_from_name("rx_30"), RXGate(math.pi / 6))
        self.assertEqual(
            self.fake_backend_target.operation_from_name("rx"),
            RXGate(self.fake_backend._theta),
        )
        self.assertEqual(self.ideal_sim_target.operation_from_name("ccx"), CCXGate())

    def test_get_instructions_for_qargs(self):
        with self.assertRaises(KeyError):
            self.empty_target.operations_for_qargs((0,))
        expected = [RZGate(self.theta), IGate(), SXGate(), XGate(), Measure()]
        res = self.ibm_target.operations_for_qargs((0,))
        for gate in expected:
            self.assertIn(gate, res)
        expected = [ECRGate()]
        res = self.fake_backend_target.operations_for_qargs((1, 0))
        for gate in expected:
            self.assertIn(gate, res)
        expected = [CXGate()]
        res = self.fake_backend_target.operations_for_qargs((0, 1))
        self.assertEqual(expected, res)
        ideal_sim_expected = [
            UGate(self.theta, self.phi, self.lam),
            RXGate(self.theta),
            RYGate(self.theta),
            RZGate(self.theta),
            CXGate(),
            ECRGate(),
            CCXGate(),
            Measure(),
        ]
        for gate in ideal_sim_expected:
            self.assertIn(gate, self.ideal_sim_target.operations_for_qargs(None))

    def test_get_operation_for_qargs_global(self):
        expected = [
            RXGate(self.theta),
            RYGate(self.theta),
            RZGate(self.theta),
            RGate(self.theta, self.phi),
            Measure(),
        ]
        res = self.aqt_target.operations_for_qargs((0,))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)
        expected = [RXXGate(self.theta)]
        res = self.aqt_target.operations_for_qargs((0, 1))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)

    def test_get_invalid_operations_for_qargs(self):
        with self.assertRaises(KeyError):
            self.ibm_target.operations_for_qargs((0, 102))
        with self.assertRaises(KeyError):
            self.ibm_target.operations_for_qargs(None)

    def test_get_operation_names_for_qargs(self):
        with self.assertRaises(KeyError):
            self.empty_target.operation_names_for_qargs((0,))
        expected = {"rz", "id", "sx", "x", "measure"}
        res = self.ibm_target.operation_names_for_qargs((0,))
        for gate in expected:
            self.assertIn(gate, res)
        expected = {"ecr"}
        res = self.fake_backend_target.operation_names_for_qargs((1, 0))
        for gate in expected:
            self.assertIn(gate, res)
        expected = {"cx"}
        res = self.fake_backend_target.operation_names_for_qargs((0, 1))
        self.assertEqual(expected, res)
        ideal_sim_expected = ["u", "rx", "ry", "rz", "cx", "ecr", "ccx", "measure"]
        for gate in ideal_sim_expected:
            self.assertIn(gate, self.ideal_sim_target.operation_names_for_qargs(None))

    def test_get_operation_names_for_qargs_invalid_qargs(self):
        with self.assertRaises(KeyError):
            self.ibm_target.operation_names_for_qargs((0, 102))
        with self.assertRaises(KeyError):
            self.ibm_target.operation_names_for_qargs(None)

    def test_get_operation_names_for_qargs_global_insts(self):
        expected = {"r", "rx", "rz", "ry", "measure"}
        self.assertEqual(self.aqt_target.operation_names_for_qargs((0,)), expected)
        expected = {
            "rxx",
        }
        self.assertEqual(self.aqt_target.operation_names_for_qargs((0, 1)), expected)

    def test_coupling_map(self):
        self.assertEqual(
            CouplingMap().get_edges(), self.empty_target.build_coupling_map().get_edges()
        )
        self.assertEqual(
            set(CouplingMap.from_full(5).get_edges()),
            set(self.aqt_target.build_coupling_map().get_edges()),
        )
        self.assertEqual(
            {(0, 1), (1, 0)}, set(self.fake_backend_target.build_coupling_map().get_edges())
        )
        self.assertEqual(
            {
                (3, 4),
                (4, 3),
                (3, 1),
                (1, 3),
                (1, 2),
                (2, 1),
                (0, 1),
                (1, 0),
            },
            set(self.ibm_target.build_coupling_map().get_edges()),
        )
        self.assertEqual(None, self.ideal_sim_target.build_coupling_map())

    def test_coupling_map_mutations_do_not_propagate(self):
        cm = CouplingMap.from_line(5, bidirectional=False)
        cx_props = {
            edge: InstructionProperties(duration=270.22e-9, error=0.00713)
            for edge in cm.get_edges()
        }
        target = Target()
        target.add_instruction(CXGate(), cx_props)
        self.assertEqual(cm, target.build_coupling_map())
        symmetric = target.build_coupling_map()
        symmetric.make_symmetric()
        self.assertNotEqual(cm, symmetric)  # sanity check for the test.
        # Verify that mutating the output of `build_coupling_map` doesn't affect the target.
        self.assertNotEqual(target.build_coupling_map(), symmetric)

    def test_coupling_map_filtered_mutations_do_not_propagate(self):
        cm = CouplingMap.from_line(5, bidirectional=False)
        cx_props = {
            edge: InstructionProperties(duration=270.22e-9, error=0.00713)
            for edge in cm.get_edges()
            if 2 not in edge
        }
        target = Target()
        target.add_instruction(CXGate(), cx_props)
        symmetric = target.build_coupling_map(filter_idle_qubits=True)
        symmetric.make_symmetric()
        self.assertNotEqual(cm, symmetric)  # sanity check for the test.
        # Verify that mutating the output of `build_coupling_map` doesn't affect the target.
        self.assertNotEqual(target.build_coupling_map(filter_idle_qubits=True), symmetric)

    def test_coupling_map_no_filter_mutations_do_not_propagate(self):
        cm = CouplingMap.from_line(5, bidirectional=False)
        cx_props = {
            edge: InstructionProperties(duration=270.22e-9, error=0.00713)
            for edge in cm.get_edges()
        }
        target = Target()
        target.add_instruction(CXGate(), cx_props)
        # The filter here does not actually do anything, because there's no idle qubits.  This is
        # just a test that this path is also not cached.
        self.assertEqual(cm, target.build_coupling_map(filter_idle_qubits=True))
        symmetric = target.build_coupling_map(filter_idle_qubits=True)
        symmetric.make_symmetric()
        self.assertNotEqual(cm, symmetric)  # sanity check for the test.
        # Verify that mutating the output of `build_coupling_map` doesn't affect the target.
        self.assertNotEqual(target.build_coupling_map(filter_idle_qubits=True), symmetric)

    def test_coupling_map_2q_gate(self):
        cmap = self.fake_backend_target.build_coupling_map("ecr")
        self.assertEqual(
            [
                (1, 0),
            ],
            cmap.get_edges(),
        )

    def test_coupling_map_3q_gate(self):
        fake_target = Target()
        ccx_props = {
            (0, 1, 2): None,
            (1, 0, 2): None,
            (2, 1, 0): None,
        }
        fake_target.add_instruction(CCXGate(), ccx_props)
        with self.assertLogs("qiskit.transpiler.target", level="WARN") as log:
            cmap = fake_target.build_coupling_map()
        self.assertEqual(
            log.output,
            [
                "WARNING:qiskit.transpiler.target:"
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. This will not be reflected in "
                "the output coupling map."
            ],
        )
        self.assertEqual([], cmap.get_edges())
        with self.assertRaises(ValueError):
            fake_target.build_coupling_map("ccx")

    def test_coupling_map_mixed_ideal_global_1q_and_2q_gates(self):
        n_qubits = 3
        target = Target()
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        target.add_instruction(RXGate(Parameter("theta")), {None: None})
        cmap = target.build_coupling_map()
        self.assertEqual([(0, 1), (1, 2)], cmap.get_edges())

    def test_coupling_map_mixed_global_1q_and_2q_gates(self):
        n_qubits = 3
        target = Target()
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        target.add_instruction(RXGate(Parameter("theta")))
        cmap = target.build_coupling_map()
        self.assertEqual([(0, 1), (1, 2)], cmap.get_edges())

    def test_coupling_map_mixed_ideal_global_2q_and_real_2q_gates(self):
        n_qubits = 3
        target = Target()
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        target.add_instruction(ECRGate())
        cmap = target.build_coupling_map()
        self.assertIsNone(cmap)

    def test_physical_qubits(self):
        self.assertEqual([], self.empty_target.physical_qubits)
        self.assertEqual(list(range(5)), self.ibm_target.physical_qubits)
        self.assertEqual(list(range(5)), self.aqt_target.physical_qubits)
        self.assertEqual(list(range(2)), self.fake_backend_target.physical_qubits)
        self.assertEqual(list(range(3)), self.ideal_sim_target.physical_qubits)

    def test_duplicate_instruction_add_instruction(self):
        target = Target()
        target.add_instruction(XGate(), {(0,): None})
        with self.assertRaises(AttributeError):
            target.add_instruction(XGate(), {(1,): None})

    def test_durations(self):
        empty_durations = self.empty_target.durations()
        self.assertEqual(
            empty_durations.duration_by_name_qubits, InstructionDurations().duration_by_name_qubits
        )
        aqt_durations = self.aqt_target.durations()
        self.assertEqual(aqt_durations.duration_by_name_qubits, {})
        ibm_durations = self.ibm_target.durations()
        expected = {
            ("cx", (0, 1)): (5.1911e-07, "s"),
            ("cx", (1, 0)): (5.5466e-07, "s"),
            ("cx", (1, 2)): (2.2755e-07, "s"),
            ("cx", (1, 3)): (4.9777e-07, "s"),
            ("cx", (2, 1)): (2.6311e-07, "s"),
            ("cx", (3, 1)): (4.6222e-07, "s"),
            ("cx", (3, 4)): (2.7022e-07, "s"),
            ("cx", (4, 3)): (3.0577e-07, "s"),
            ("id", (0,)): (3.55e-08, "s"),
            ("id", (1,)): (3.55e-08, "s"),
            ("id", (2,)): (3.55e-08, "s"),
            ("id", (3,)): (3.55e-08, "s"),
            ("id", (4,)): (3.55e-08, "s"),
            ("measure", (0,)): (5.813e-06, "s"),
            ("measure", (1,)): (5.813e-06, "s"),
            ("measure", (2,)): (5.813e-06, "s"),
            ("measure", (3,)): (5.813e-06, "s"),
            ("measure", (4,)): (5.813e-06, "s"),
            ("rz", (0,)): (0, "s"),
            ("rz", (1,)): (0, "s"),
            ("rz", (2,)): (0, "s"),
            ("rz", (3,)): (0, "s"),
            ("rz", (4,)): (0, "s"),
            ("sx", (0,)): (3.55e-08, "s"),
            ("sx", (1,)): (3.55e-08, "s"),
            ("sx", (2,)): (3.55e-08, "s"),
            ("sx", (3,)): (3.55e-08, "s"),
            ("sx", (4,)): (3.55e-08, "s"),
            ("x", (0,)): (3.55e-08, "s"),
            ("x", (1,)): (3.55e-08, "s"),
            ("x", (2,)): (3.55e-08, "s"),
            ("x", (3,)): (3.55e-08, "s"),
            ("x", (4,)): (3.55e-08, "s"),
        }
        self.assertEqual(ibm_durations.duration_by_name_qubits, expected)

    def test_mapping(self):
        with self.assertRaises(KeyError):
            _res = self.empty_target["cx"]
        expected = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.assertEqual(self.aqt_target["r"], expected)
        self.assertEqual(["rx", "ry", "rz", "r", "rxx", "measure"], list(self.aqt_target))
        expected_values = [
            {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            {
                (0, 1): None,
                (0, 2): None,
                (0, 3): None,
                (0, 4): None,
                (1, 0): None,
                (2, 0): None,
                (3, 0): None,
                (4, 0): None,
                (1, 2): None,
                (1, 3): None,
                (1, 4): None,
                (2, 1): None,
                (3, 1): None,
                (4, 1): None,
                (2, 3): None,
                (2, 4): None,
                (3, 2): None,
                (4, 2): None,
                (3, 4): None,
                (4, 3): None,
            },
            {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
        ]
        self.assertEqual(expected_values, list(self.aqt_target.values()))
        expected_items = {
            "rx": {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            "ry": {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            "rz": {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            "r": {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
            "rxx": {
                (0, 1): None,
                (0, 2): None,
                (0, 3): None,
                (0, 4): None,
                (1, 0): None,
                (2, 0): None,
                (3, 0): None,
                (4, 0): None,
                (1, 2): None,
                (1, 3): None,
                (1, 4): None,
                (2, 1): None,
                (3, 1): None,
                (4, 1): None,
                (2, 3): None,
                (2, 4): None,
                (3, 2): None,
                (4, 2): None,
                (3, 4): None,
                (4, 3): None,
            },
            "measure": {
                (0,): None,
                (1,): None,
                (2,): None,
                (3,): None,
                (4,): None,
            },
        }
        self.assertEqual(expected_items, dict(self.aqt_target.items()))
        self.assertIn("cx", self.ibm_target)
        self.assertNotIn("ecr", self.ibm_target)
        self.assertEqual(len(self.ibm_target), 6)

    def test_update_instruction_properties(self):
        self.aqt_target.update_instruction_properties(
            "rxx",
            (0, 1),
            InstructionProperties(duration=1e-6, error=1e-5),
        )
        self.assertEqual(self.aqt_target["rxx"][(0, 1)].duration, 1e-6)
        self.assertEqual(self.aqt_target["rxx"][(0, 1)].error, 1e-5)

    def test_update_instruction_properties_invalid_instruction(self):
        with self.assertRaises(KeyError):
            self.ibm_target.update_instruction_properties("rxx", (0, 1), None)

    def test_update_instruction_properties_invalid_qarg(self):
        with self.assertRaises(KeyError):
            self.fake_backend_target.update_instruction_properties("ecr", (0, 1), None)

    def test_str(self):
        expected = """Target
Number of qubits: 5
Instructions:
	id
		(0,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000413
		(1,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000502
		(2,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.0004003
		(3,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000614
		(4,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.006149
	rz
		(0,):
			Duration: 0 sec.
			Error Rate: 0
		(1,):
			Duration: 0 sec.
			Error Rate: 0
		(2,):
			Duration: 0 sec.
			Error Rate: 0
		(3,):
			Duration: 0 sec.
			Error Rate: 0
		(4,):
			Duration: 0 sec.
			Error Rate: 0
	sx
		(0,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000413
		(1,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000502
		(2,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.0004003
		(3,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000614
		(4,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.006149
	x
		(0,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000413
		(1,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000502
		(2,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.0004003
		(3,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000614
		(4,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.006149
	cx
		(3, 4):
			Duration: 2.7022e-07 sec.
			Error Rate: 0.00713
		(4, 3):
			Duration: 3.0577e-07 sec.
			Error Rate: 0.00713
		(3, 1):
			Duration: 4.6222e-07 sec.
			Error Rate: 0.00929
		(1, 3):
			Duration: 4.9777e-07 sec.
			Error Rate: 0.00929
		(1, 2):
			Duration: 2.2755e-07 sec.
			Error Rate: 0.00659
		(2, 1):
			Duration: 2.6311e-07 sec.
			Error Rate: 0.00659
		(0, 1):
			Duration: 5.1911e-07 sec.
			Error Rate: 0.01201
		(1, 0):
			Duration: 5.5466e-07 sec.
			Error Rate: 0.01201
	measure
		(0,):
			Duration: 5.813e-06 sec.
			Error Rate: 0.0751
		(1,):
			Duration: 5.813e-06 sec.
			Error Rate: 0.0225
		(2,):
			Duration: 5.813e-06 sec.
			Error Rate: 0.0146
		(3,):
			Duration: 5.813e-06 sec.
			Error Rate: 0.0215
		(4,):
			Duration: 5.813e-06 sec.
			Error Rate: 0.0333
"""
        self.assertEqual(expected, str(self.ibm_target))
        aqt_expected = """Target: AQT Target
Number of qubits: 5
Instructions:
	rx
		(0,)
		(1,)
		(2,)
		(3,)
		(4,)
	ry
		(0,)
		(1,)
		(2,)
		(3,)
		(4,)
	rz
		(0,)
		(1,)
		(2,)
		(3,)
		(4,)
	r
		(0,)
		(1,)
		(2,)
		(3,)
		(4,)
	rxx
		(0, 1)
		(0, 2)
		(0, 3)
		(0, 4)
		(1, 0)
		(2, 0)
		(3, 0)
		(4, 0)
		(1, 2)
		(1, 3)
		(1, 4)
		(2, 1)
		(3, 1)
		(4, 1)
		(2, 3)
		(2, 4)
		(3, 2)
		(4, 2)
		(3, 4)
		(4, 3)
	measure
		(0,)
		(1,)
		(2,)
		(3,)
		(4,)
"""
        self.assertEqual(aqt_expected, str(self.aqt_target))
        sim_expected = """Target: Ideal Simulator
Number of qubits: 3
Instructions:
	u
	rx
	ry
	rz
	cx
	ecr
	ccx
	measure
"""
        self.assertEqual(sim_expected, str(self.ideal_sim_target))

    def test_extra_props_str(self):
        target = Target(description="Extra Properties")

        class ExtraProperties(InstructionProperties):
            """An example properties subclass."""

            def __init__(
                self,
                duration=None,
                error=None,
                calibration=None,
                tuned=None,
                diamond_norm_error=None,
            ):
                super().__init__(duration=duration, error=error, calibration=calibration)
                self.tuned = tuned
                self.diamond_norm_error = diamond_norm_error

        cx_props = {
            (3, 4): ExtraProperties(
                duration=270.22e-9, error=0.00713, tuned=False, diamond_norm_error=2.12e-6
            ),
        }
        target.add_instruction(CXGate(), cx_props)
        expected = """Target: Extra Properties
Number of qubits: 5
Instructions:
	cx
		(3, 4):
			Duration: 2.7022e-07 sec.
			Error Rate: 0.00713
"""
        self.assertEqual(expected, str(target))

    def test_timing_constraints(self):
        generated_constraints = self.aqt_target.timing_constraints()
        expected_constraints = TimingConstraints()
        for i in ["granularity", "min_length", "pulse_alignment", "acquire_alignment"]:
            self.assertEqual(
                getattr(generated_constraints, i),
                getattr(expected_constraints, i),
                f"Generated constraints differs from expected for attribute {i}"
                f"{getattr(generated_constraints, i)}!={getattr(expected_constraints, i)}",
            )

    def test_get_non_global_operation_name_ideal_backend(self):
        self.assertEqual(self.aqt_target.get_non_global_operation_names(), [])
        self.assertEqual(self.ideal_sim_target.get_non_global_operation_names(), [])
        self.assertEqual(self.ibm_target.get_non_global_operation_names(), [])
        self.assertEqual(self.fake_backend_target.get_non_global_operation_names(), [])

    def test_get_non_global_operation_name_ideal_backend_strict_direction(self):
        self.assertEqual(self.aqt_target.get_non_global_operation_names(True), [])
        self.assertEqual(self.ideal_sim_target.get_non_global_operation_names(True), [])
        self.assertEqual(self.ibm_target.get_non_global_operation_names(True), [])
        self.assertEqual(
            self.fake_backend_target.get_non_global_operation_names(True), ["cx", "ecr"]
        )

    def test_instruction_supported(self):
        self.assertTrue(self.aqt_target.instruction_supported("r", (0,)))
        self.assertFalse(self.aqt_target.instruction_supported("cx", (0, 1)))
        self.assertTrue(self.ideal_sim_target.instruction_supported("cx", (0, 1)))
        self.assertFalse(self.ideal_sim_target.instruction_supported("cx", (0, 524)))
        self.assertTrue(self.fake_backend_target.instruction_supported("cx", (0, 1)))
        self.assertFalse(self.fake_backend_target.instruction_supported("cx", (1, 0)))
        self.assertFalse(self.ideal_sim_target.instruction_supported("cx", (0, 1, 2)))

    def test_instruction_supported_parameters(self):
        mumbai = FakeMumbaiFractionalCX()
        self.assertTrue(
            mumbai.target.instruction_supported(
                qargs=(0, 1), operation_class=RZXGate, parameters=[math.pi / 4]
            )
        )
        self.assertTrue(mumbai.target.instruction_supported(qargs=(0, 1), operation_class=RZXGate))
        self.assertTrue(
            mumbai.target.instruction_supported(operation_class=RZXGate, parameters=[math.pi / 4])
        )
        self.assertFalse(mumbai.target.instruction_supported("rzx", parameters=[math.pi / 4]))
        self.assertTrue(mumbai.target.instruction_supported("rz", parameters=[Parameter("angle")]))
        self.assertTrue(
            mumbai.target.instruction_supported("rzx_45", qargs=(0, 1), parameters=[math.pi / 4])
        )
        self.assertTrue(mumbai.target.instruction_supported("rzx_45", qargs=(0, 1)))
        self.assertTrue(mumbai.target.instruction_supported("rzx_45", parameters=[math.pi / 4]))
        self.assertFalse(mumbai.target.instruction_supported("rzx_45", parameters=[math.pi / 6]))
        self.assertFalse(
            mumbai.target.instruction_supported("rzx_45", parameters=[Parameter("angle")])
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported(
                qargs=(0,), operation_class=RXGate, parameters=[Parameter("angle")]
            )
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported(
                qargs=(0,), operation_class=RXGate, parameters=[math.pi]
            )
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported(
                operation_class=RXGate, parameters=[math.pi]
            )
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported(
                operation_class=RXGate, parameters=[Parameter("angle")]
            )
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported(
                "rx", qargs=(0,), parameters=[Parameter("angle")]
            )
        )
        self.assertTrue(
            self.ideal_sim_target.instruction_supported("rx", qargs=(0,), parameters=[math.pi])
        )
        self.assertTrue(self.ideal_sim_target.instruction_supported("rx", parameters=[math.pi]))
        self.assertTrue(
            self.ideal_sim_target.instruction_supported("rx", parameters=[Parameter("angle")])
        )

    def test_instruction_supported_multiple_parameters(self):
        target = Target(1)
        target.add_instruction(
            UGate(self.theta, self.phi, self.lam),
            {(0,): InstructionProperties(duration=270.22e-9, error=0.00713)},
        )
        self.assertFalse(target.instruction_supported("u", parameters=[math.pi]))
        self.assertTrue(target.instruction_supported("u", parameters=[math.pi, math.pi, math.pi]))
        self.assertTrue(
            target.instruction_supported(
                operation_class=UGate, parameters=[math.pi, math.pi, math.pi]
            )
        )
        self.assertFalse(
            target.instruction_supported(operation_class=UGate, parameters=[Parameter("x")])
        )

    def test_instruction_supported_arg_len_mismatch(self):
        self.assertFalse(
            self.ideal_sim_target.instruction_supported(operation_class=UGate, parameters=[math.pi])
        )
        self.assertFalse(self.ideal_sim_target.instruction_supported("u", parameters=[math.pi]))

    def test_instruction_supported_class_not_in_target(self):
        self.assertFalse(
            self.ibm_target.instruction_supported(operation_class=CZGate, parameters=[math.pi])
        )

    def test_instruction_supported_no_args(self):
        self.assertFalse(self.ibm_target.instruction_supported())

    def test_instruction_supported_no_operation(self):
        self.assertFalse(self.ibm_target.instruction_supported(qargs=(0,), parameters=[math.pi]))


class TestPulseTarget(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.pulse_target = Target(
            dt=3e-7, granularity=2, min_length=4, pulse_alignment=8, acquire_alignment=8
        )
        with pulse.build(name="sx_q0") as self.custom_sx_q0:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))
        with pulse.build(name="sx_q1") as self.custom_sx_q1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(1))
        sx_props = {
            (0,): InstructionProperties(
                duration=35.5e-9, error=0.000413, calibration=self.custom_sx_q0
            ),
            (1,): InstructionProperties(
                duration=35.5e-9, error=0.000502, calibration=self.custom_sx_q1
            ),
        }
        self.pulse_target.add_instruction(SXGate(), sx_props)

    def test_instruction_schedule_map(self):
        inst_map = self.pulse_target.instruction_schedule_map()
        self.assertIn("sx", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction("sx"), [0, 1])
        self.assertTrue("sx" in inst_map.qubit_instructions(0))

    def test_instruction_schedule_map_ideal_sim_backend(self):
        ideal_sim_target = Target(num_qubits=3)
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lambda")
        for inst in [
            UGate(theta, phi, lam),
            RXGate(theta),
            RYGate(theta),
            RZGate(theta),
            CXGate(),
            ECRGate(),
            CCXGate(),
            Measure(),
        ]:
            ideal_sim_target.add_instruction(inst, {None: None})
        inst_map = ideal_sim_target.instruction_schedule_map()
        self.assertEqual(InstructionScheduleMap(), inst_map)

    def test_str(self):
        expected = """Target
Number of qubits: 2
Instructions:
	sx
		(0,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000413
			With pulse schedule calibration
		(1,):
			Duration: 3.55e-08 sec.
			Error Rate: 0.000502
			With pulse schedule calibration
"""
        self.assertEqual(expected, str(self.pulse_target))

    def test_update_from_instruction_schedule_map_add_instruction(self):
        target = Target()
        inst_map = InstructionScheduleMap()
        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, self.custom_sx_q1)
        target.update_from_instruction_schedule_map(inst_map, {"sx": SXGate()})
        self.assertEqual(inst_map, target.instruction_schedule_map())

    def test_update_from_instruction_schedule_map_with_schedule_parameter(self):
        self.pulse_target.dt = None
        inst_map = InstructionScheduleMap()
        duration = Parameter("duration")

        with pulse.build(name="sx_q0") as custom_sx:
            pulse.play(pulse.Constant(duration, 0.2), pulse.DriveChannel(0))

        inst_map.add("sx", 0, custom_sx, ["duration"])

        target = Target(dt=3e-7)
        target.update_from_instruction_schedule_map(inst_map, {"sx": SXGate()})
        self.assertEqual(inst_map, target.instruction_schedule_map())

    def test_update_from_instruction_schedule_map_update_schedule(self):
        self.pulse_target.dt = None
        inst_map = InstructionScheduleMap()
        with pulse.build(name="sx_q1") as custom_sx:
            pulse.play(pulse.Constant(1000, 0.2), pulse.DriveChannel(1))

        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, custom_sx)
        self.pulse_target.update_from_instruction_schedule_map(inst_map, {"sx": SXGate()})
        self.assertEqual(inst_map, self.pulse_target.instruction_schedule_map())
        # Calibration doesn't change for q0
        self.assertEqual(self.pulse_target["sx"][(0,)].duration, 35.5e-9)
        self.assertEqual(self.pulse_target["sx"][(0,)].error, 0.000413)
        # Calibration is updated for q1 without error dict and gate time
        self.assertIsNone(self.pulse_target["sx"][(1,)].duration)
        self.assertIsNone(self.pulse_target["sx"][(1,)].error)

    def test_update_from_instruction_schedule_map_new_instruction_no_name_map(self):
        target = Target()
        inst_map = InstructionScheduleMap()
        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, self.custom_sx_q1)
        target.update_from_instruction_schedule_map(inst_map)
        self.assertEqual(target["sx"][(0,)].calibration, self.custom_sx_q0)
        self.assertEqual(target["sx"][(1,)].calibration, self.custom_sx_q1)

    def test_update_from_instruction_schedule_map_new_qarg_raises(self):
        inst_map = InstructionScheduleMap()
        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, self.custom_sx_q1)
        inst_map.add("sx", 2, self.custom_sx_q1)
        self.pulse_target.update_from_instruction_schedule_map(inst_map)
        self.assertFalse(self.pulse_target.instruction_supported("sx", (2,)))

    def test_update_from_instruction_schedule_map_with_dt_set(self):
        inst_map = InstructionScheduleMap()
        with pulse.build(name="sx_q1") as custom_sx:
            pulse.play(pulse.Constant(1000, 0.2), pulse.DriveChannel(1))

        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, custom_sx)
        self.pulse_target.dt = 1.0
        self.pulse_target.update_from_instruction_schedule_map(inst_map, {"sx": SXGate()})
        self.assertEqual(inst_map, self.pulse_target.instruction_schedule_map())
        self.assertEqual(self.pulse_target["sx"][(1,)].duration, 1000.0)
        self.assertIsNone(self.pulse_target["sx"][(1,)].error)
        # This is an edge case.
        # System dt is read-only property and changing it will break all underlying calibrations.
        # duration of sx0 returns previous value since calibration doesn't change.
        self.assertEqual(self.pulse_target["sx"][(0,)].duration, 35.5e-9)
        self.assertEqual(self.pulse_target["sx"][(0,)].error, 0.000413)

    def test_update_from_instruction_schedule_map_with_error_dict(self):
        inst_map = InstructionScheduleMap()
        with pulse.build(name="sx_q1") as custom_sx:
            pulse.play(pulse.Constant(1000, 0.2), pulse.DriveChannel(1))

        inst_map.add("sx", 0, self.custom_sx_q0)
        inst_map.add("sx", 1, custom_sx)
        self.pulse_target.dt = 1.0
        error_dict = {"sx": {(1,): 1.0}}

        self.pulse_target.update_from_instruction_schedule_map(
            inst_map, {"sx": SXGate()}, error_dict=error_dict
        )
        self.assertEqual(self.pulse_target["sx"][(1,)].error, 1.0)
        self.assertEqual(self.pulse_target["sx"][(0,)].error, 0.000413)

    def test_timing_constraints(self):
        generated_constraints = self.pulse_target.timing_constraints()
        expected_constraints = TimingConstraints(2, 4, 8, 8)
        for i in ["granularity", "min_length", "pulse_alignment", "acquire_alignment"]:
            self.assertEqual(
                getattr(generated_constraints, i),
                getattr(expected_constraints, i),
                f"Generated constraints differs from expected for attribute {i}"
                f"{getattr(generated_constraints, i)}!={getattr(expected_constraints, i)}",
            )

    def test_default_instmap_has_no_custom_gate(self):
        backend = FakeGeneva()
        target = backend.target

        # This copies .calibraiton of InstructionProperties of each instruction
        # This must not convert PulseQobj to Schedule during this.
        # See qiskit-terra/#9595
        inst_map = target.instruction_schedule_map()
        self.assertFalse(inst_map.has_custom_gate())

        # Get pulse schedule. This generates Schedule provided by backend.
        sched = inst_map.get("sx", (0,))
        self.assertEqual(sched.metadata["publisher"], CalibrationPublisher.BACKEND_PROVIDER)
        self.assertFalse(inst_map.has_custom_gate())

        # Update target with custom instruction. This is user provided schedule.
        new_prop = InstructionProperties(
            duration=self.custom_sx_q0.duration,
            error=None,
            calibration=self.custom_sx_q0,
        )
        target.update_instruction_properties(instruction="sx", qargs=(0,), properties=new_prop)
        inst_map = target.instruction_schedule_map()
        self.assertTrue(inst_map.has_custom_gate())

        empty = InstructionProperties()
        target.update_instruction_properties(instruction="sx", qargs=(0,), properties=empty)
        inst_map = target.instruction_schedule_map()
        self.assertFalse(inst_map.has_custom_gate())

    def test_get_empty_target_calibration(self):
        target = Target()
        properties = {(0,): InstructionProperties(duration=100, error=0.1)}
        target.add_instruction(XGate(), properties)

        self.assertIsNone(target["x"][(0,)].calibration)

    def test_loading_legacy_ugate_instmap(self):
        # This is typical IBM backend situation.
        # IBM provider used to have u1, u2, u3 in the basis gates and
        # these have been replaced with sx and rz.
        # However, IBM provider still provides calibration of these u gates,
        # and the inst map loads them as backend calibrations.
        # Target is implicitly updated with inst map when it is set in transpile.
        # If u gates are not excluded, they may appear in the transpiled circuit.
        # These gates are no longer supported by hardware.
        entry = ScheduleDef()
        entry.define(pulse.Schedule(name="fake_u3"), user_provided=False)  # backend provided
        instmap = InstructionScheduleMap()
        instmap._add("u3", (0,), entry)

        # Today's standard IBM backend target with sx, rz basis
        target = Target()
        target.add_instruction(SXGate(), {(0,): InstructionProperties()})
        target.add_instruction(RZGate(Parameter("θ")), {(0,): InstructionProperties()})
        target.add_instruction(Measure(), {(0,): InstructionProperties()})
        names_before = set(target.operation_names)

        target.update_from_instruction_schedule_map(instmap)
        names_after = set(target.operation_names)

        # Otherwise u3 and sx-rz basis conflict in 1q decomposition.
        self.assertSetEqual(names_before, names_after)


class TestGlobalVariableWidthOperations(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.theta = Parameter("theta")
        self.phi = Parameter("phi")
        self.lam = Parameter("lambda")
        self.target_global_gates_only = Target(num_qubits=5)
        self.target_global_gates_only.add_instruction(CXGate())
        self.target_global_gates_only.add_instruction(UGate(self.theta, self.phi, self.lam))
        self.target_global_gates_only.add_instruction(Measure())
        self.target_global_gates_only.add_instruction(IfElseOp, name="if_else")
        self.target_global_gates_only.add_instruction(ForLoopOp, name="for_loop")
        self.target_global_gates_only.add_instruction(WhileLoopOp, name="while_loop")
        self.target_global_gates_only.add_instruction(SwitchCaseOp, name="switch_case")
        self.ibm_target = Target()
        i_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(IGate(), i_props)
        rz_props = {
            (0,): InstructionProperties(duration=0, error=0),
            (1,): InstructionProperties(duration=0, error=0),
            (2,): InstructionProperties(duration=0, error=0),
            (3,): InstructionProperties(duration=0, error=0),
            (4,): InstructionProperties(duration=0, error=0),
        }
        self.ibm_target.add_instruction(RZGate(self.theta), rz_props)
        sx_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(SXGate(), sx_props)
        x_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
            (2,): InstructionProperties(duration=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(duration=35.5e-9, error=0.000614),
            (4,): InstructionProperties(duration=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(XGate(), x_props)
        cx_props = {
            (3, 4): InstructionProperties(duration=270.22e-9, error=0.00713),
            (4, 3): InstructionProperties(duration=305.77e-9, error=0.00713),
            (3, 1): InstructionProperties(duration=462.22e-9, error=0.00929),
            (1, 3): InstructionProperties(duration=497.77e-9, error=0.00929),
            (1, 2): InstructionProperties(duration=227.55e-9, error=0.00659),
            (2, 1): InstructionProperties(duration=263.11e-9, error=0.00659),
            (0, 1): InstructionProperties(duration=519.11e-9, error=0.01201),
            (1, 0): InstructionProperties(duration=554.66e-9, error=0.01201),
        }
        self.ibm_target.add_instruction(CXGate(), cx_props)
        measure_props = {
            (0,): InstructionProperties(duration=5.813e-6, error=0.0751),
            (1,): InstructionProperties(duration=5.813e-6, error=0.0225),
            (2,): InstructionProperties(duration=5.813e-6, error=0.0146),
            (3,): InstructionProperties(duration=5.813e-6, error=0.0215),
            (4,): InstructionProperties(duration=5.813e-6, error=0.0333),
        }
        self.ibm_target.add_instruction(Measure(), measure_props)
        self.ibm_target.add_instruction(IfElseOp, name="if_else")
        self.ibm_target.add_instruction(ForLoopOp, name="for_loop")
        self.ibm_target.add_instruction(WhileLoopOp, name="while_loop")
        self.ibm_target.add_instruction(SwitchCaseOp, name="switch_case")
        self.aqt_target = Target(description="AQT Target")
        rx_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RXGate(self.theta), rx_props)
        ry_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RYGate(self.theta), ry_props)
        rz_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RZGate(self.theta), rz_props)
        r_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(RGate(self.theta, self.phi), r_props)
        rxx_props = {
            (0, 1): None,
            (0, 2): None,
            (0, 3): None,
            (0, 4): None,
            (1, 0): None,
            (2, 0): None,
            (3, 0): None,
            (4, 0): None,
            (1, 2): None,
            (1, 3): None,
            (1, 4): None,
            (2, 1): None,
            (3, 1): None,
            (4, 1): None,
            (2, 3): None,
            (2, 4): None,
            (3, 2): None,
            (4, 2): None,
            (3, 4): None,
            (4, 3): None,
        }
        self.aqt_target.add_instruction(RXXGate(self.theta), rxx_props)
        measure_props = {
            (0,): None,
            (1,): None,
            (2,): None,
            (3,): None,
            (4,): None,
        }
        self.aqt_target.add_instruction(Measure(), measure_props)
        self.aqt_target.add_instruction(IfElseOp, name="if_else")
        self.aqt_target.add_instruction(ForLoopOp, name="for_loop")
        self.aqt_target.add_instruction(WhileLoopOp, name="while_loop")
        self.aqt_target.add_instruction(SwitchCaseOp, name="switch_case")

    def test_qargs(self):
        expected_ibm = {
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (3, 4),
            (4, 3),
            (3, 1),
            (1, 3),
            (1, 2),
            (2, 1),
            (0, 1),
            (1, 0),
        }
        self.assertEqual(expected_ibm, self.ibm_target.qargs)
        expected_aqt = {
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (3, 1),
            (4, 1),
            (2, 3),
            (2, 4),
            (3, 2),
            (4, 2),
            (3, 4),
            (4, 3),
        }
        self.assertEqual(expected_aqt, self.aqt_target.qargs)
        self.assertEqual(None, self.target_global_gates_only.qargs)

    def test_qargs_single_qarg(self):
        target = Target()
        target.add_instruction(XGate(), {(0,): None})
        self.assertEqual(
            {
                (0,),
            },
            target.qargs,
        )

    def test_qargs_for_operation_name(self):
        self.assertEqual(
            self.ibm_target.qargs_for_operation_name("rz"), {(0,), (1,), (2,), (3,), (4,)}
        )
        self.assertEqual(
            self.aqt_target.qargs_for_operation_name("rz"), {(0,), (1,), (2,), (3,), (4,)}
        )
        self.assertIsNone(self.target_global_gates_only.qargs_for_operation_name("cx"))
        self.assertIsNone(self.ibm_target.qargs_for_operation_name("if_else"))
        self.assertIsNone(self.aqt_target.qargs_for_operation_name("while_loop"))
        self.assertIsNone(self.aqt_target.qargs_for_operation_name("switch_case"))

    def test_instruction_names(self):
        self.assertEqual(
            self.ibm_target.operation_names,
            {
                "rz",
                "id",
                "sx",
                "x",
                "cx",
                "measure",
                "if_else",
                "while_loop",
                "for_loop",
                "switch_case",
            },
        )
        self.assertEqual(
            self.aqt_target.operation_names,
            {
                "rz",
                "ry",
                "rx",
                "rxx",
                "r",
                "measure",
                "if_else",
                "while_loop",
                "for_loop",
                "switch_case",
            },
        )
        self.assertEqual(
            self.target_global_gates_only.operation_names,
            {"u", "cx", "measure", "if_else", "while_loop", "for_loop", "switch_case"},
        )

    def test_operations_for_qargs(self):
        expected = [
            IGate(),
            RZGate(self.theta),
            SXGate(),
            XGate(),
            Measure(),
            IfElseOp,
            ForLoopOp,
            WhileLoopOp,
            SwitchCaseOp,
        ]
        res = self.ibm_target.operations_for_qargs((0,))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)
        expected = [
            CXGate(),
            IfElseOp,
            ForLoopOp,
            WhileLoopOp,
            SwitchCaseOp,
        ]
        res = self.ibm_target.operations_for_qargs((0, 1))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)
        expected = [
            RXGate(self.theta),
            RYGate(self.theta),
            RZGate(self.theta),
            RGate(self.theta, self.phi),
            Measure(),
            IfElseOp,
            ForLoopOp,
            WhileLoopOp,
            SwitchCaseOp,
        ]
        res = self.aqt_target.operations_for_qargs((0,))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)
        expected = [RXXGate(self.theta), IfElseOp, ForLoopOp, WhileLoopOp, SwitchCaseOp]
        res = self.aqt_target.operations_for_qargs((0, 1))
        self.assertEqual(len(expected), len(res))
        for x in expected:
            self.assertIn(x, res)

    def test_operation_names_for_qargs(self):
        expected = {
            "id",
            "rz",
            "sx",
            "x",
            "measure",
            "if_else",
            "for_loop",
            "while_loop",
            "switch_case",
        }
        self.assertEqual(expected, self.ibm_target.operation_names_for_qargs((0,)))
        expected = {
            "cx",
            "if_else",
            "for_loop",
            "while_loop",
            "switch_case",
        }
        self.assertEqual(expected, self.ibm_target.operation_names_for_qargs((0, 1)))
        expected = {
            "rx",
            "ry",
            "rz",
            "r",
            "measure",
            "if_else",
            "for_loop",
            "while_loop",
            "switch_case",
        }
        self.assertEqual(self.aqt_target.operation_names_for_qargs((0,)), expected)
        expected = {"rxx", "if_else", "for_loop", "while_loop", "switch_case"}
        self.assertEqual(self.aqt_target.operation_names_for_qargs((0, 1)), expected)

    def test_operations(self):
        ibm_expected = [
            RZGate(self.theta),
            IGate(),
            SXGate(),
            XGate(),
            CXGate(),
            Measure(),
            WhileLoopOp,
            IfElseOp,
            ForLoopOp,
            SwitchCaseOp,
        ]
        for gate in ibm_expected:
            self.assertIn(gate, self.ibm_target.operations)
        aqt_expected = [
            RZGate(self.theta),
            RXGate(self.theta),
            RYGate(self.theta),
            RGate(self.theta, self.phi),
            RXXGate(self.theta),
            ForLoopOp,
            IfElseOp,
            WhileLoopOp,
            SwitchCaseOp,
        ]
        for gate in aqt_expected:
            self.assertIn(gate, self.aqt_target.operations)
        fake_expected = [
            UGate(self.theta, self.phi, self.lam),
            CXGate(),
            Measure(),
            ForLoopOp,
            WhileLoopOp,
            IfElseOp,
            SwitchCaseOp,
        ]
        for gate in fake_expected:
            self.assertIn(gate, self.target_global_gates_only.operations)

    def test_add_invalid_instruction(self):
        inst_props = {(0, 1, 2, 3): None}
        target = Target()
        with self.assertRaises(TranspilerError):
            target.add_instruction(CXGate(), inst_props)

    def test_instructions(self):
        ibm_expected = [
            (IGate(), (0,)),
            (IGate(), (1,)),
            (IGate(), (2,)),
            (IGate(), (3,)),
            (IGate(), (4,)),
            (RZGate(self.theta), (0,)),
            (RZGate(self.theta), (1,)),
            (RZGate(self.theta), (2,)),
            (RZGate(self.theta), (3,)),
            (RZGate(self.theta), (4,)),
            (SXGate(), (0,)),
            (SXGate(), (1,)),
            (SXGate(), (2,)),
            (SXGate(), (3,)),
            (SXGate(), (4,)),
            (XGate(), (0,)),
            (XGate(), (1,)),
            (XGate(), (2,)),
            (XGate(), (3,)),
            (XGate(), (4,)),
            (CXGate(), (3, 4)),
            (CXGate(), (4, 3)),
            (CXGate(), (3, 1)),
            (CXGate(), (1, 3)),
            (CXGate(), (1, 2)),
            (CXGate(), (2, 1)),
            (CXGate(), (0, 1)),
            (CXGate(), (1, 0)),
            (Measure(), (0,)),
            (Measure(), (1,)),
            (Measure(), (2,)),
            (Measure(), (3,)),
            (Measure(), (4,)),
            (IfElseOp, None),
            (ForLoopOp, None),
            (WhileLoopOp, None),
            (SwitchCaseOp, None),
        ]
        self.assertEqual(ibm_expected, self.ibm_target.instructions)
        ideal_sim_expected = [
            (CXGate(), None),
            (UGate(self.theta, self.phi, self.lam), None),
            (Measure(), None),
            (IfElseOp, None),
            (ForLoopOp, None),
            (WhileLoopOp, None),
            (SwitchCaseOp, None),
        ]
        self.assertEqual(ideal_sim_expected, self.target_global_gates_only.instructions)

    def test_instruction_supported(self):
        self.assertTrue(self.aqt_target.instruction_supported("r", (0,)))
        self.assertFalse(self.aqt_target.instruction_supported("cx", (0, 1)))
        self.assertTrue(self.target_global_gates_only.instruction_supported("cx", (0, 1)))
        self.assertFalse(self.target_global_gates_only.instruction_supported("cx", (0, 524)))
        self.assertFalse(self.target_global_gates_only.instruction_supported("cx", (0, 1, 2)))
        self.assertTrue(self.aqt_target.instruction_supported("while_loop", (0, 1, 2, 3)))
        self.assertTrue(
            self.aqt_target.instruction_supported(operation_class=WhileLoopOp, qargs=(0, 1, 2, 3))
        )
        self.assertTrue(
            self.aqt_target.instruction_supported(operation_class=SwitchCaseOp, qargs=(0, 1, 2, 3))
        )
        self.assertFalse(
            self.ibm_target.instruction_supported(
                operation_class=IfElseOp, qargs=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            )
        )
        self.assertFalse(
            self.ibm_target.instruction_supported(operation_class=IfElseOp, qargs=(0, 425))
        )
        self.assertFalse(self.ibm_target.instruction_supported("for_loop", qargs=(0, 425)))

    def test_coupling_map(self):
        self.assertIsNone(self.target_global_gates_only.build_coupling_map())
        self.assertEqual(
            set(CouplingMap.from_full(5).get_edges()),
            set(self.aqt_target.build_coupling_map().get_edges()),
        )
        self.assertEqual(
            {
                (3, 4),
                (4, 3),
                (3, 1),
                (1, 3),
                (1, 2),
                (2, 1),
                (0, 1),
                (1, 0),
            },
            set(self.ibm_target.build_coupling_map().get_edges()),
        )

    def test_mixed_ideal_target_filtered_coupling_map(self):
        target = Target(num_qubits=10)
        target.add_instruction(
            XGate(), {(qubit,): InstructionProperties(error=0.5) for qubit in range(5)}
        )
        target.add_instruction(
            CXGate(),
            {
                edge: InstructionProperties(error=0.6)
                for edge in CouplingMap.from_line(5, bidirectional=False).get_edges()
            },
        )
        target.add_instruction(SXGate())
        coupling_map = target.build_coupling_map(filter_idle_qubits=True)
        self.assertEqual(max(coupling_map.physical_qubits), 4)
        self.assertEqual(coupling_map.get_edges(), [(0, 1), (1, 2), (2, 3), (3, 4)])


class TestInstructionProperties(QiskitTestCase):
    def test_empty_repr(self):
        properties = InstructionProperties()
        self.assertEqual(
            repr(properties),
            "InstructionProperties(duration=None, error=None, calibration=None)",
        )


class TestTargetFromConfiguration(QiskitTestCase):
    """Test the from_configuration() constructor."""

    def test_basis_gates_qubits_only(self):
        """Test construction with only basis gates."""
        target = Target.from_configuration(["u", "cx"], 3)
        self.assertEqual(target.operation_names, {"u", "cx"})

    def test_basis_gates_no_qubits(self):
        target = Target.from_configuration(["u", "cx"])
        self.assertEqual(target.operation_names, {"u", "cx"})

    def test_basis_gates_coupling_map(self):
        """Test construction with only basis gates."""
        target = Target.from_configuration(
            ["u", "cx"], 3, CouplingMap.from_ring(3, bidirectional=False)
        )
        self.assertEqual(target.operation_names, {"u", "cx"})
        self.assertEqual({(0,), (1,), (2,)}, target["u"].keys())
        self.assertEqual({(0, 1), (1, 2), (2, 0)}, target["cx"].keys())

    def test_properties(self):
        fake_backend = FakeVigo()
        config = fake_backend.configuration()
        properties = fake_backend.properties()
        target = Target.from_configuration(
            basis_gates=config.basis_gates,
            num_qubits=config.num_qubits,
            coupling_map=CouplingMap(config.coupling_map),
            backend_properties=properties,
        )
        self.assertEqual(0, target["rz"][(0,)].error)
        self.assertEqual(0, target["rz"][(0,)].duration)

    def test_properties_with_durations(self):
        fake_backend = FakeVigo()
        config = fake_backend.configuration()
        properties = fake_backend.properties()
        durations = InstructionDurations([("rz", 0, 0.5)], dt=1.0)
        target = Target.from_configuration(
            basis_gates=config.basis_gates,
            num_qubits=config.num_qubits,
            coupling_map=CouplingMap(config.coupling_map),
            backend_properties=properties,
            instruction_durations=durations,
            dt=config.dt,
        )
        self.assertEqual(0.5, target["rz"][(0,)].duration)

    def test_inst_map(self):
        fake_backend = FakeNairobi()
        config = fake_backend.configuration()
        properties = fake_backend.properties()
        defaults = fake_backend.defaults()
        constraints = TimingConstraints(**config.timing_constraints)
        target = Target.from_configuration(
            basis_gates=config.basis_gates,
            num_qubits=config.num_qubits,
            coupling_map=CouplingMap(config.coupling_map),
            backend_properties=properties,
            dt=config.dt,
            inst_map=defaults.instruction_schedule_map,
            timing_constraints=constraints,
        )
        self.assertIsNotNone(target["sx"][(0,)].calibration)
        self.assertEqual(target.granularity, constraints.granularity)
        self.assertEqual(target.min_length, constraints.min_length)
        self.assertEqual(target.pulse_alignment, constraints.pulse_alignment)
        self.assertEqual(target.acquire_alignment, constraints.acquire_alignment)

    def test_concurrent_measurements(self):
        fake_backend = FakeVigo()
        config = fake_backend.configuration()
        target = Target.from_configuration(
            basis_gates=config.basis_gates,
            concurrent_measurements=config.meas_map,
        )
        self.assertEqual(target.concurrent_measurements, config.meas_map)

    def test_custom_basis_gates(self):
        basis_gates = ["my_x", "cx"]
        custom_name_mapping = {"my_x": XGate()}
        target = Target.from_configuration(
            basis_gates=basis_gates, num_qubits=2, custom_name_mapping=custom_name_mapping
        )
        self.assertEqual(target.operation_names, {"my_x", "cx"})

    def test_missing_custom_basis_no_coupling(self):
        basis_gates = ["my_X", "cx"]
        with self.assertRaisesRegex(KeyError, "is not present in the standard gate names"):
            Target.from_configuration(basis_gates, num_qubits=4)

    def test_missing_custom_basis_with_coupling(self):
        basis_gates = ["my_X", "cx"]
        cmap = CouplingMap.from_line(3)
        with self.assertRaisesRegex(KeyError, "is not present in the standard gate names"):
            Target.from_configuration(basis_gates, 3, cmap)

    def test_over_two_qubit_gate_without_coupling(self):
        basis_gates = ["ccx", "cx", "swap", "u"]
        target = Target.from_configuration(basis_gates, 15)
        self.assertEqual(target.operation_names, {"ccx", "cx", "swap", "u"})

    def test_over_two_qubits_with_coupling(self):
        basis_gates = ["ccx", "cx", "swap", "u"]
        cmap = CouplingMap.from_line(15)
        with self.assertRaisesRegex(TranspilerError, "This constructor method only supports"):
            Target.from_configuration(basis_gates, 15, cmap)
