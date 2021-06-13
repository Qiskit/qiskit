# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the InstructionScheduleMap."""
import numpy as np

import qiskit.pulse.library as library
from qiskit.circuit.library.standard_gates import U1Gate, U3Gate, CXGate, XGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse import (
    InstructionScheduleMap,
    Play,
    PulseError,
    Schedule,
    ScheduleBlock,
    Waveform,
    ShiftPhase,
    Constant,
)
from qiskit.pulse.channels import DriveChannel
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class TestInstructionScheduleMap(QiskitTestCase):
    """Test the InstructionScheduleMap."""

    def test_add(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u1", 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

        with self.assertRaises(PulseError):
            inst_map.add("u1", (), sched)
        with self.assertRaises(PulseError):
            inst_map.add("u1", 1, "not a schedule")

    def test_add_block(self):
        """Test add block, and that errors are raised when expected."""
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u1", 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

    def test_instructions(self):
        """Test `instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u3", 0, sched)

        instructions = inst_map.instructions
        for inst in ["u1", "u3"]:
            self.assertTrue(inst in instructions)

    def test_has(self):
        """Test `has` and `assert_has`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertTrue(inst_map.has("u1", [0]))
        self.assertTrue(inst_map.has("cx", (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has("cx", 100)

    def test_has_from_mock(self):
        """Test `has` and `assert_has` from mock data."""
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedule_map
        self.assertTrue(inst_map.has("u1", [0]))
        self.assertTrue(inst_map.has("cx", (0, 1)))
        self.assertTrue(inst_map.has("u3", 0))
        self.assertTrue(inst_map.has("measure", [0, 1]))
        self.assertFalse(inst_map.has("u1", [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has("cx", 100)

    def test_qubits_with_instruction(self):
        """Test `qubits_with_instruction`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("u1", (1,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction("cx"), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction("none"), [])

    def test_qubit_instructions(self):
        """Test `qubit_instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("u1", (1,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertEqual(inst_map.qubit_instructions(0), ["u1"])
        self.assertEqual(inst_map.qubit_instructions(1), ["u1"])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ["cx"])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add("x", 0, sched)

        self.assertEqual(sched, inst_map.get("x", (0,)))

    def test_get_block(self):
        """Test `get` block."""
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add("x", 0, sched)

        self.assertEqual(sched, inst_map.get("x", (0,)))

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 0, sched)
        inst_map.remove("tmp", 0)
        self.assertFalse(inst_map.has("tmp", 0))
        with self.assertRaises(PulseError):
            inst_map.remove("not_there", (0,))
        self.assertFalse("tmp" in inst_map.qubit_instructions(0))

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 100, sched)
        self.assertEqual(inst_map.pop("tmp", 100), sched)
        self.assertFalse(inst_map.has("tmp", 100))

        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction("tmp"), [])
        with self.assertRaises(PulseError):
            inst_map.pop("not_there", (0,))

    def test_add_gate(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U1Gate(0), 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), (), sched)
        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), 1, "not a schedule")

    def test_instructions_gate(self):
        """Test `instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U3Gate(0, 0, 0), 0, sched)

        instructions = inst_map.instructions
        for inst in ["u1", "u3"]:
            self.assertTrue(inst in instructions)

    def test_has_gate(self):
        """Test `has` and `assert_has`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_has_from_mock_gate(self):
        """Test `has` and `assert_has` from mock data."""
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedule_map
        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        self.assertTrue(inst_map.has(U3Gate(0, 0, 0), 0))
        self.assertTrue(inst_map.has("measure", [0, 1]))
        self.assertFalse(inst_map.has(U1Gate(0), [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_qubits_with_instruction_gate(self):
        """Test `qubits_with_instruction`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction(CXGate()), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction("none"), [])

    def test_qubit_instructions_gate(self):
        """Test `qubit_instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertEqual(inst_map.qubit_instructions(0), ["u1"])
        self.assertEqual(inst_map.qubit_instructions(1), ["u1"])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ["cx"])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get_gate(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()
        inst_map.add(XGate(), 0, sched)

        self.assertEqual(sched, inst_map.get(XGate(), (0,)))

    def test_remove_gate(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 0, sched)
        inst_map.remove("tmp", 0)
        self.assertFalse(inst_map.has("tmp", 0))
        with self.assertRaises(PulseError):
            inst_map.remove("not_there", (0,))
        self.assertFalse("tmp" in inst_map.qubit_instructions(0))

    def test_pop_gate(self):
        """Test pop with default."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(XGate(), 100, sched)
        self.assertEqual(inst_map.pop(XGate(), 100), sched)
        self.assertFalse(inst_map.has(XGate(), 100))

        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction(XGate()), [])
        with self.assertRaises(PulseError):
            inst_map.pop("not_there", (0,))

    def test_sequenced_parameterized_schedule(self):
        """Test parameterized schedule consists of multiple instruction."""

        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [
            PulseQobjInstruction(name="fc", ch="d0", t0=10, phase="P1"),
            PulseQobjInstruction(name="fc", ch="d0", t0=20, phase="P2"),
            PulseQobjInstruction(name="fc", ch="d0", t0=30, phase="P3"),
        ]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        inst_map = InstructionScheduleMap()

        inst_map.add("inst_seq", 0, Schedule(*converted_instruction, name="inst_seq"))

        with self.assertRaises(PulseError):
            inst_map.get("inst_seq", 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            inst_map.get("inst_seq", 0, 1, 2, 3, 4, 5, 6, 7, 8)

        p3_expr = Parameter("p3")
        p3_expr = p3_expr.bind({p3_expr: 3})

        sched = inst_map.get("inst_seq", 0, 1, 2, p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get("inst_seq", 0, P1=1, P2=2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get("inst_seq", 0, 1, 2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

    def test_schedule_generator(self):
        """Test schedule generator functionalty."""

        dur_val = 10
        amp = 1.0

        def test_func(dur: int):
            sched = Schedule()
            sched += Play(library.constant(int(dur), amp), DriveChannel(0))
            return sched

        expected_sched = Schedule()
        expected_sched += Play(library.constant(dur_val, amp), DriveChannel(0))

        inst_map = InstructionScheduleMap()
        inst_map.add("f", (0,), test_func)
        self.assertEqual(inst_map.get("f", (0,), dur_val), expected_sched)

        self.assertEqual(inst_map.get_parameters("f", (0,)), ("dur",))

    def test_schedule_generator_supports_parameter_expressions(self):
        """Test expression-based schedule generator functionalty."""

        t_param = Parameter("t")
        amp = 1.0

        def test_func(dur: ParameterExpression, t_val: int):
            dur_bound = dur.bind({t_param: t_val})
            sched = Schedule()
            sched += Play(library.constant(int(float(dur_bound)), amp), DriveChannel(0))
            return sched

        expected_sched = Schedule()
        expected_sched += Play(library.constant(10, amp), DriveChannel(0))

        inst_map = InstructionScheduleMap()
        inst_map.add("f", (0,), test_func)
        self.assertEqual(inst_map.get("f", (0,), dur=2 * t_param, t_val=5), expected_sched)

        self.assertEqual(
            inst_map.get_parameters("f", (0,)),
            (
                "dur",
                "t_val",
            ),
        )

    def test_schedule_with_non_alphanumeric_ordering(self):
        """Test adding and getting schedule with non obvious parameter ordering."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        lamb = Parameter("lambda")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(theta, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(phi, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(lamb, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched, arguments=["theta", "phi", "lambda"])

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(0, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(2, DriveChannel(0)), inplace=True)

        # if parameter is alphanumerical ordering this maps to
        # theta -> 2
        # phi -> 1
        # lamb -> 0
        # however non alphanumerical ordering is specified in add method thus mapping should be
        # theta -> 0
        # phi -> 1
        # lamb -> 2
        test_sched = inst_map.get("target_sched", (0,), 0, 1, 2)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertEqual(test_inst[1], ref_inst[1])

    def test_binding_too_many_parameters(self):
        """Test getting schedule with too many parameter binding."""
        param = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        with self.assertRaises(PulseError):
            inst_map.get("target_sched", (0,), 0, 1, 2, 3)

    def test_binding_unassigned_parameters(self):
        """Test getting schedule with unassigned parameter binding."""
        param = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        with self.assertRaises(PulseError):
            inst_map.get("target_sched", (0,), P0=0)

    def test_schedule_with_multiple_parameters_under_same_name(self):
        """Test getting schedule with parameters that have the same name."""
        param1 = Parameter("param")
        param2 = Parameter("param")
        param3 = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(param3, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(1.23, DriveChannel(0)), inplace=True)

        test_sched = inst_map.get("target_sched", (0,), param=1.23)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_get_schedule_with_unbound_parameter(self):
        """Test get schedule with partial binding."""
        param1 = Parameter("param1")
        param2 = Parameter("param2")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)

        test_sched = inst_map.get("target_sched", (0,), param2=1.23)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_partially_bound_callable(self):
        """Test register partial function."""
        import functools

        def callable_schedule(par_b, par_a):
            sched = Schedule()
            sched.insert(10, Play(Constant(10, par_b), DriveChannel(0)), inplace=True)
            sched.insert(20, Play(Constant(10, par_a), DriveChannel(0)), inplace=True)
            return sched

        ref_sched = Schedule()
        ref_sched.insert(10, Play(Constant(10, 0.1), DriveChannel(0)), inplace=True)
        ref_sched.insert(20, Play(Constant(10, 0.2), DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()

        def test_callable_sched1(par_b):
            return callable_schedule(par_b, 0.2)

        inst_map.add("my_gate1", (0,), test_callable_sched1, ["par_b"])
        ret_sched = inst_map.get("my_gate1", (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)

        # bind partially
        test_callable_sched2 = functools.partial(callable_schedule, par_a=0.2)

        inst_map.add("my_gate2", (0,), test_callable_sched2, ["par_b"])
        ret_sched = inst_map.get("my_gate2", (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)
