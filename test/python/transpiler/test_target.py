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

import numpy as np

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
)
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameter import Parameter
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler import Target
from qiskit.transpiler import InstructionProperties
from qiskit.test import QiskitTestCase
from qiskit.test.mock.fake_backend_v2 import FakeBackendV2


class TestTarget(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.fake_backend = FakeBackendV2()
        self.fake_backend_target = self.fake_backend.target
        self.theta = Parameter("theta")
        self.phi = Parameter("phi")
        self.ibm_target = Target()
        i_props = {
            (0,): InstructionProperties(length=35.5e-9, error=0.000413),
            (1,): InstructionProperties(length=35.5e-9, error=0.000502),
            (2,): InstructionProperties(length=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(length=35.5e-9, error=0.000614),
            (4,): InstructionProperties(length=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(IGate(), i_props)
        rz_props = {
            (0,): InstructionProperties(length=0, error=0),
            (1,): InstructionProperties(length=0, error=0),
            (2,): InstructionProperties(length=0, error=0),
            (3,): InstructionProperties(length=0, error=0),
            (4,): InstructionProperties(length=0, error=0),
        }
        self.ibm_target.add_instruction(RZGate(self.theta), rz_props)
        sx_props = {
            (0,): InstructionProperties(length=35.5e-9, error=0.000413),
            (1,): InstructionProperties(length=35.5e-9, error=0.000502),
            (2,): InstructionProperties(length=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(length=35.5e-9, error=0.000614),
            (4,): InstructionProperties(length=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(SXGate(), sx_props)
        x_props = {
            (0,): InstructionProperties(length=35.5e-9, error=0.000413),
            (1,): InstructionProperties(length=35.5e-9, error=0.000502),
            (2,): InstructionProperties(length=35.5e-9, error=0.0004003),
            (3,): InstructionProperties(length=35.5e-9, error=0.000614),
            (4,): InstructionProperties(length=35.5e-9, error=0.006149),
        }
        self.ibm_target.add_instruction(XGate(), x_props)
        cx_props = {
            (3, 4): InstructionProperties(length=270.22e-9, error=0.00713),
            (4, 3): InstructionProperties(length=305.77e-9, error=0.00713),
            (3, 1): InstructionProperties(length=462.22e-9, error=0.00929),
            (1, 3): InstructionProperties(length=497.77e-9, error=0.00929),
            (1, 2): InstructionProperties(length=227.55e-9, error=0.00659),
            (2, 1): InstructionProperties(length=263.11e-9, error=0.00659),
            (0, 1): InstructionProperties(length=519.11e-9, error=0.01201),
            (1, 0): InstructionProperties(length=554.66e-9, error=0.01201),
        }
        self.ibm_target.add_instruction(CXGate(), cx_props)
        measure_props = {
            (0,): InstructionProperties(length=5.813e-6, error=0.0751),
            (1,): InstructionProperties(length=5.813e-6, error=0.0225),
            (2,): InstructionProperties(length=5.813e-6, error=0.0146),
            (3,): InstructionProperties(length=5.813e-6, error=0.0215),
            (4,): InstructionProperties(length=5.813e-6, error=0.0333),
        }
        self.ibm_target.add_instruction(Measure(), measure_props)

        self.aqt_target = Target()
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

    def test_get_qargs(self):
        with self.assertRaises(KeyError):
            self.empty_target.get_qargs("rz")
        self.assertEqual(self.ibm_target.get_qargs("rz"), {(0,), (1,), (2,), (3,), (4,)})
        self.assertEqual(self.aqt_target.get_qargs("rz"), {(0,), (1,), (2,), (3,), (4,)})
        self.assertEqual(self.fake_backend_target.get_qargs("cx"), {(0, 1), (1, 0)})
        self.assertEqual(
            self.fake_backend_target.get_qargs("ecr"),
            {
                (1, 0),
            },
        )

    def test_instruction_names(self):
        self.assertEqual(self.empty_target.instruction_names, set())
        self.assertEqual(
            self.ibm_target.instruction_names, {"rz", "id", "sx", "x", "cx", "measure"}
        )
        self.assertEqual(
            self.aqt_target.instruction_names, {"rz", "ry", "rx", "rxx", "r", "measure"}
        )
        self.assertEqual(
            self.fake_backend_target.instruction_names, {"u", "cx", "measure", "ecr", "rx_30", "rx"}
        )

    def test_instructions(self):
        self.assertEqual(self.empty_target.instructions, [])
        ibm_expected = [RZGate(self.theta), IGate(), SXGate(), XGate(), CXGate(), Measure()]
        for gate in ibm_expected:
            self.assertIn(gate, self.ibm_target.instructions)
        aqt_expected = [
            RZGate(self.theta),
            RXGate(self.theta),
            RYGate(self.theta),
            RGate(self.theta, self.phi),
            RXXGate(self.theta),
        ]
        for gate in aqt_expected:
            self.assertIn(gate, self.aqt_target.instructions)
        fake_expected = [
            UGate(self.fake_backend._theta, self.fake_backend._phi, self.fake_backend._lam),
            CXGate(),
            Measure(),
            ECRGate(),
            RXGate(math.pi / 6),
            RXGate(self.fake_backend._theta),
        ]
        for gate in fake_expected:
            self.assertIn(gate, self.fake_backend_target.instructions)

    def test_get_gates_from_name(self):
        with self.assertRaises(KeyError):
            self.empty_target.get_gate_from_name("measure")
        self.assertEqual(self.ibm_target.get_gate_from_name("measure"), Measure())
        self.assertEqual(self.fake_backend_target.get_gate_from_name("rx_30"), RXGate(math.pi / 6))
        self.assertEqual(
            self.fake_backend_target.get_gate_from_name("rx"), RXGate(self.fake_backend._theta)
        )

    def test_get_instructions_for_qargs(self):
        with self.assertRaises(KeyError):
            self.empty_target.get_instructions_for_qargs((0,))
        expected = [RZGate(self.theta), IGate(), SXGate(), XGate(), Measure()]
        res = self.ibm_target.get_instructions_for_qargs((0,))
        for gate in expected:
            self.assertIn(gate, res)
        expected = [CXGate(), ECRGate()]
        res = self.fake_backend_target.get_instructions_for_qargs((1, 0))
        for gate in expected:
            self.assertIn(gate, res)
        expected = [CXGate()]
        res = self.fake_backend_target.get_instructions_for_qargs((0, 1))
        self.assertEqual(expected, res)

    def test_coupling_map(self):
        self.assertEqual(CouplingMap().get_edges(), self.empty_target.coupling_map().get_edges())
        self.assertEqual(
            set(CouplingMap.from_full(5).get_edges()),
            set(self.aqt_target.coupling_map().get_edges()),
        )
        self.assertEqual({(0, 1), (1, 0)}, set(self.fake_backend_target.coupling_map().get_edges()))
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
            set(self.ibm_target.coupling_map().get_edges()),
        )

    def test_coupling_map_2q_gate(self):
        cmap = self.fake_backend_target.coupling_map("ecr")
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
            cmap = fake_target.coupling_map()
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
            fake_target.coupling_map("ccx")

    def test_distance_matrix_no_weight(self):
        expected = np.array(
            [
                [
                    0,
                    1,
                ],
                [1, 0],
            ]
        )
        self.assertTrue(np.array_equal(self.fake_backend_target.distance_matrix(), expected))
        self.assertEqual(self.empty_target.distance_matrix().size, 0)
        expected = np.array(
            [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]
        )
        self.assertTrue(np.array_equal(self.aqt_target.distance_matrix(), expected))
        expected = np.array(
            [
                [0, 1, 2, 2, 3],
                [1, 0, 1, 1, 2],
                [2, 1, 0, 2, 3],
                [2, 1, 2, 0, 1],
                [3, 2, 3, 1, 0],
            ]
        )
        self.assertTrue(np.array_equal(self.ibm_target.distance_matrix(), expected))

    def test_distance_matrix_error(self):
        expected = np.zeros((5, 5))
        self.assertTrue(np.array_equal(self.aqt_target.distance_matrix("error"), expected))
        expected = np.array(
            [
                [0.0, 0.01201, 0.0186, 0.0213, 0.02843],
                [0.01201, 0.0, 0.00659, 0.00929, 0.01642],
                [0.0186, 0.00659, 0.0, 0.01588, 0.02301],
                [0.0213, 0.00929, 0.01588, 0.0, 0.00713],
                [0.02843, 0.01642, 0.02301, 0.00713, 0.0],
            ]
        )
        self.assertTrue(np.allclose(self.ibm_target.distance_matrix("error"), expected))

    def test_distance_matrix_length(self):
        expected = np.zeros((5, 5))
        self.assertTrue(np.array_equal(self.aqt_target.distance_matrix("length"), expected))
        expected = np.array(
            [
                [0.00000e00, 5.19110e-07, 7.46660e-07, 9.81330e-07, 1.25155e-06],
                [5.19110e-07, 0.00000e00, 2.27550e-07, 4.62220e-07, 7.32440e-07],
                [7.46660e-07, 2.27550e-07, 0.00000e00, 6.89770e-07, 9.59990e-07],
                [9.81330e-07, 4.62220e-07, 6.89770e-07, 0.00000e00, 2.70220e-07],
                [1.25155e-06, 7.32440e-07, 9.59990e-07, 2.70220e-07, 0.00000e00],
            ]
        )
        self.assertTrue(np.allclose(self.ibm_target.distance_matrix("length"), expected))

    def test_distance_matrix_3q(self):
        fake_target = Target()
        ccx_props = {
            (0, 1, 2): None,
            (1, 0, 2): None,
            (2, 1, 0): None,
        }
        fake_target.add_instruction(CCXGate(), ccx_props)
        with self.assertLogs("qiskit.transpiler.target", level="WARN") as log:
            mat = fake_target.distance_matrix()
        self.assertEqual(
            log.output,
            [
                "WARNING:qiskit.transpiler.target:"
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. These gates will not be reflected in "
                "the output matrix."
            ],
        )
        self.assertTrue(np.array_equal(np.zeros((3, 3)), mat))

    def test_physical_qubits(self):
        self.assertEqual([], self.empty_target.physical_qubits)
        self.assertEqual(list(range(5)), self.ibm_target.physical_qubits)
        self.assertEqual(list(range(5)), self.aqt_target.physical_qubits)
        self.assertEqual(list(range(2)), self.fake_backend_target.physical_qubits)

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


class TestInstructionProperties(QiskitTestCase):
    def test_empty_repr(self):
        properties = InstructionProperties()
        self.assertEqual(
            repr(properties),
            "InstructionProperties(length=None, error=None, pulse=None, properties=None)",
        )
