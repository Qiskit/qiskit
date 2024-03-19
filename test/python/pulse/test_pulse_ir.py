# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse IR"""
import copy

from test import QiskitTestCase
from rustworkx import is_isomorphic_node_match

from qiskit.pulse import (
    Constant,
    Play,
    Delay,
    ShiftPhase,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.transforms import AlignLeft, AlignRight
from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.exceptions import PulseError


class TestSequenceIR(QiskitTestCase):
    """Test SequenceIR objects"""

    def test_ir_creation(self):
        """Test ir creation"""
        ir_example = SequenceIR(AlignLeft())
        self.assertEqual(ir_example.sequence.num_nodes(), 2)
        self.assertEqual(ir_example.initial_time(), None)
        self.assertEqual(ir_example.final_time(), None)
        self.assertEqual(ir_example.duration, None)

    def test_add_elements(self):
        """Test addition of elements"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)

        self.assertEqual(ir_example.sequence.num_nodes(), 3)
        self.assertEqual(ir_example.elements()[0], inst)

        inst = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        ir_example.append(inst)
        self.assertEqual(ir_example.sequence.num_nodes(), 4)
        self.assertEqual(len(ir_example.elements()), 2)
        self.assertEqual(ir_example.elements()[1], inst)

    def test_initial_time(self):
        """Test initial time"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example._time_table[2] = 100
        ir_example._time_table[3] = 200
        # Just for the sake of the test. The minimal initial time has to have an edge with 0.
        ir_example._time_table[4] = 50
        ir_example._sequence.add_edge(0, 2, None)
        ir_example._sequence.add_edge(0, 3, None)
        self.assertEqual(ir_example.initial_time(), 100)

        ir_example._time_table[3] = 0
        self.assertEqual(ir_example.initial_time(), 0)

    def test_initial_time_partial_scheduling(self):
        """Test initial time with partial scheduling"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example._time_table[2] = 100
        ir_example._time_table[3] = None
        ir_example._sequence.add_edge(0, 2, None)
        ir_example._sequence.add_edge(0, 3, None)
        self.assertEqual(ir_example.initial_time(), None)

    def test_final_time(self):
        """Test final time"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example.append(inst)
        # Just for the sake of the test. The maximal final time has to have an edge with 1.
        ir_example._time_table[2] = 1000
        ir_example._time_table[3] = 100
        ir_example._time_table[4] = 200
        ir_example._sequence.add_edge(3, 1, None)
        ir_example._sequence.add_edge(4, 1, None)
        self.assertEqual(ir_example.final_time(), 300)

    def test_final_time_partial_scheduling(self):
        """Test final time with partial scheduling"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example._time_table[2] = 1000
        ir_example._time_table[3] = None
        ir_example._sequence.add_edge(2, 1, None)
        ir_example._sequence.add_edge(3, 1, None)
        self.assertEqual(ir_example.final_time(), None)

    def test_duration(self):
        """Test duration"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example._time_table[2] = 100
        ir_example._time_table[3] = 300
        ir_example._sequence.add_edge(0, 2, None)
        ir_example._sequence.add_edge(3, 1, None)

        self.assertEqual(ir_example.initial_time(), 100)
        self.assertEqual(ir_example.final_time(), 400)
        self.assertEqual(ir_example.duration, 300)

    def test_duration_with_sub_block(self):
        """Test duration with sub block"""
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        sub_block = SequenceIR(AlignLeft())
        sub_block.append(inst)
        sub_block._time_table[2] = 0
        sub_block._sequence.add_edge(0, 2, None)
        sub_block._sequence.add_edge(2, 1, None)

        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.duration, 100)

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(inst)
        ir_example.append(sub_block)
        ir_example._time_table[2] = 100
        ir_example._time_table[3] = 300
        ir_example._sequence.add_edge(0, 2, None)
        ir_example._sequence.add_edge(3, 1, None)

        self.assertEqual(ir_example.initial_time(), 100)
        self.assertEqual(ir_example.final_time(), 400)
        self.assertEqual(ir_example.duration, 300)

    def test_inst_targets_no_sub_blocks(self):
        """Test that inst targets are recovered correctly with no sub blocks"""
        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), frame=QubitFrame(1), target=Qubit(1)))
        ir_example.append(Play(Constant(100, 0.1), frame=QubitFrame(2), target=Qubit(2)))
        ir_example.append(Delay(100, target=Qubit(3)))
        ir_example.append(ShiftPhase(100, frame=QubitFrame(3)))

        # TODO : Make sure this also works for acquire.

        inst_targets = ir_example.inst_targets
        ref = {
            MixedFrame(Qubit(1), QubitFrame(1)),
            MixedFrame(Qubit(2), QubitFrame(2)),
            Qubit(3),
            QubitFrame(3),
        }

        self.assertEqual(inst_targets, ref)

    def test_inst_targets_with_sub_blocks(self):
        """Test that inst targets are recovered correctly with sub blocks"""
        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), frame=QubitFrame(1), target=Qubit(1)))

        sub_block = SequenceIR(AlignLeft())
        sub_block.append(Delay(100, target=Qubit(1)))
        ir_example.append(sub_block)

        inst_targets = ir_example.inst_targets
        ref = {MixedFrame(Qubit(1), QubitFrame(1)), Qubit(1)}

        self.assertEqual(inst_targets, ref)

    def test_scheduled_elements_sort(self):
        """Test that scheduled elements is sorted correctly"""
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        ir_example = SequenceIR(AlignRight())
        ir_example.append(inst)
        ir_example.append(inst)
        ir_example.append(inst)

        ir_example._time_table[2] = 200
        ir_example._time_table[3] = 100
        ir_example._time_table[4] = 500

        sch_elements = ir_example.scheduled_elements()
        self.assertEqual(len(sch_elements), 3)
        self.assertEqual(sch_elements[0], (100, inst))
        self.assertEqual(sch_elements[1], (200, inst))
        self.assertEqual(sch_elements[2], (500, inst))

    def test_scheduled_elements_no_recursion(self):
        """Test that scheduled elements with no recursion works"""
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        sub_block = SequenceIR(AlignLeft())
        sub_block.append(inst)

        ir_example = SequenceIR(AlignRight())
        ir_example.append(inst)
        ir_example.append(sub_block)

        sch_elements = ir_example.scheduled_elements()
        self.assertEqual(len(sch_elements), 2)
        self.assertEqual(sch_elements[0], (None, inst))
        self.assertEqual(sch_elements[1], (None, sub_block))

    def test_scheduled_elements_with_recursion(self):
        """Test that scheduled elements with recursion works"""
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        sub_block2 = SequenceIR(AlignLeft())
        sub_block2.append(inst2)
        sub_block2._time_table[2] = 0

        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        sub_block1 = SequenceIR(AlignLeft())
        sub_block1.append(inst1)
        sub_block1.append(sub_block2)
        sub_block1._time_table[2] = 0
        sub_block1._time_table[3] = 100

        inst = Play(Constant(100, 0.5), frame=QubitFrame(3), target=Qubit(3))
        ir_example = SequenceIR(AlignRight())
        ir_example.append(inst)
        ir_example.append(sub_block1)
        ir_example._time_table[2] = 0
        ir_example._time_table[3] = 100

        sch_elements = ir_example.scheduled_elements(recursive=True)
        self.assertEqual(len(sch_elements), 3)
        self.assertTrue((0, inst) in sch_elements)
        self.assertTrue((100, inst1) in sch_elements)
        self.assertTrue((200, inst2) in sch_elements)

    def test_scheduled_elements_with_recursion_raises_error(self):
        """Test that scheduled elements with recursion raises error if sub block is not scheduled"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        sub_block1 = SequenceIR(AlignLeft())
        sub_block1.append(inst1)

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(sub_block1)

        with self.assertRaises(PulseError):
            ir_example.scheduled_elements(recursive=True)

    def test_ir_equating_different_alignment(self):
        """Test equating of blocks with different alignment"""
        self.assertFalse(SequenceIR(AlignLeft()) == SequenceIR(AlignRight()))

    def test_ir_equating_different_instructions(self):
        """Test equating of blocks with different instructions"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(1))

        ir1 = SequenceIR(AlignLeft())
        ir1.append(inst1)
        ir2 = SequenceIR(AlignLeft())
        ir2.append(inst2)
        self.assertFalse(ir1 == ir2)

    def test_ir_equating_different_ordering(self):
        """Test equating of blocks with different ordering, but the same sequence structure"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))

        ir1 = SequenceIR(AlignLeft())
        ir1.append(inst1)
        ir1.append(inst2)

        ir2 = SequenceIR(AlignLeft())
        ir2.append(inst2)
        ir2.append(inst1)

        self.assertTrue(ir1 == ir2)

        ir1.sequence.add_edge(0, 2, None)
        ir1.sequence.add_edge(0, 3, None)
        ir1.sequence.add_edge(3, 1, None)
        ir1.sequence.add_edge(2, 1, None)

        ir2.sequence.add_edge(0, 2, None)
        ir2.sequence.add_edge(0, 3, None)
        ir2.sequence.add_edge(3, 1, None)
        ir2.sequence.add_edge(2, 1, None)

        self.assertTrue(ir1 == ir2)

    def test_ir_dedicated_copy(self):
        """Test the dedicated semi-deep copy method"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignRight())
        block.append(inst1)
        ir1 = SequenceIR(AlignLeft())
        ir1.append(block)
        ir1.append(inst2)
        ir1.sequence.add_edge(0, 2, None)
        ir1._time_table[3] = 100

        copied = ir1.copy()
        # Top level properties and nested IRs are new objects
        self.assertEqual(copied, ir1)
        self.assertIsNot(copied, ir1)
        self.assertEqual(copied.alignment, ir1.alignment)
        self.assertEqual(copied._time_table, ir1._time_table)
        self.assertIsNot(copied._time_table, ir1._time_table)
        # PyDAG has no built-in equality check
        self.assertTrue(
            is_isomorphic_node_match(copied._sequence, ir1._sequence, lambda x, y: x == y)
        )
        self.assertIsNot(copied._sequence, ir1._sequence)
        self.assertEqual(copied.elements()[0], ir1.elements()[0])
        self.assertIsNot(copied.elements()[0], ir1.elements()[0])
        # Instructions are passed by reference
        self.assertIs(copied.elements()[1], inst2)
        self.assertIs(copied.elements()[0].elements()[0], inst1)

    def test_ir_deepcopy(self):
        """Test the deep copy method"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignRight())
        block.append(inst1)
        ir1 = SequenceIR(AlignLeft())
        ir1.append(block)
        ir1.append(inst2)
        ir1.sequence.add_edge(0, 2, None)
        ir1._time_table[3] = 100

        copied = copy.deepcopy(ir1)
        self.assertEqual(copied, ir1)
        self.assertIsNot(copied, ir1)
        self.assertEqual(copied.alignment, ir1.alignment)
        self.assertEqual(copied._time_table, ir1._time_table)
        self.assertIsNot(copied._time_table, ir1._time_table)
        # PyDAG has no built-in equality check
        self.assertTrue(
            is_isomorphic_node_match(copied._sequence, ir1._sequence, lambda x, y: x == y)
        )
        self.assertIsNot(copied._sequence, ir1._sequence)
        self.assertEqual(copied.elements()[0], ir1.elements()[0])
        self.assertIsNot(copied.elements()[0], ir1.elements()[0])
        self.assertEqual(copied.elements()[1], inst2)
        self.assertIsNot(copied.elements()[1], inst2)
        self.assertEqual(copied.elements()[0].elements()[0], inst1)
        self.assertIsNot(copied.elements()[0].elements()[0], inst1)

    # TODO : Test SequenceIR.draw()
