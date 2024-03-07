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

from test import QiskitTestCase
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

    # TODO : Replace tests which relied on temporary passes with proper passes.
    # def test_flatten_ir_no_sub_blocks(self):
    #     """Test that flattening ir with no sub blocks doesn't do anything"""
    #     ir_example = SequenceIR(AlignLeft())
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
    #     ir_example.append(inst)
    #     ir_example.append(inst2)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     self.assertEqual(ir_example.flatten(), ir_example)
    #
    # def test_flatten_inplace_flag(self):
    #     """Test that inplace flag in flattening works"""
    #     ir_example = SequenceIR(AlignLeft())
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
    #     ir_example.append(inst)
    #     ir_example.append(inst2)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     self.assertTrue(ir_example.flatten(inplace=True) is ir_example)
    #     self.assertFalse(ir_example.flatten() is ir_example)
    #
    # def test_flatten_one_sub_block(self):
    #     """Test that flattening works with one block"""
    #     ir_example = SequenceIR(AlignLeft())
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
    #     block = SequenceIR(AlignLeft())
    #     block.append(inst)
    #     block.append(inst2)
    #     ir_example.append(block)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     flat = ir_example.flatten()
    #     edge_list = flat.sequence.edge_list()
    #     print(edge_list)
    #     self.assertEqual(len(edge_list), 4)
    #     self.assertTrue((0, 5) in edge_list)
    #     self.assertTrue((0, 6) in edge_list)
    #     self.assertTrue((5, 1) in edge_list)
    #     self.assertTrue((6, 1) in edge_list)
    #
    # def test_flatten_one_sub_block_and_parallel_instruction(self):
    #     """Test that flattening works with one block and parallel instruction"""
    #     ir_example = SequenceIR(AlignLeft())
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
    #     block = SequenceIR(AlignLeft())
    #     block.append(inst)
    #     block.append(inst2)
    #     ir_example.append(block)
    #     ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(3), target=Qubit(3)))
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     flat = ir_example.flatten()
    #     edge_list = flat.sequence.edge_list()
    #     self.assertEqual(len(edge_list), 6)
    #     self.assertTrue((0, 3) in edge_list)
    #     self.assertTrue((3, 1) in edge_list)
    #     self.assertTrue((0, 6) in edge_list)
    #     self.assertTrue((0, 7) in edge_list)
    #     self.assertTrue((6, 1) in edge_list)
    #     self.assertTrue((7, 1) in edge_list)
    #
    # def test_flatten_one_sub_block_and_sequential_instructions(self):
    #     """Test that flattening works with one block and sequential instructions"""
    #     ir_example = SequenceIR(AlignLeft())
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
    #     block = SequenceIR(AlignLeft())
    #     block.append(inst)
    #     block.append(inst2)
    #     ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1)))
    #     ir_example.append(block)
    #     ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2)))
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     flat = ir_example.flatten()
    #     edge_list = flat.sequence.edge_list()
    #     self.assertEqual(len(edge_list), 8)
    #     self.assertTrue((0, 2) in edge_list)
    #     self.assertTrue((4, 1) in edge_list)
    #     self.assertTrue((2, 7) in edge_list)
    #     self.assertTrue((2, 8) in edge_list)
    #     self.assertTrue((7, 1) in edge_list)
    #     self.assertTrue((8, 1) in edge_list)
    #     self.assertTrue((7, 4) in edge_list)
    #     self.assertTrue((8, 4) in edge_list)
    #
    # def test_flatten_two_sub_blocks(self):
    #     """Test that flattening works with two sub blocks"""
    #     inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #     inst2 = Play(Constant(200, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #
    #     block1 = SequenceIR(AlignLeft())
    #     block1.append(inst1)
    #     block2 = SequenceIR(AlignLeft())
    #     block2.append(inst2)
    #
    #     ir_example = SequenceIR(AlignLeft())
    #     ir_example.append(block1)
    #     ir_example.append(block2)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     flat = ir_example.flatten()
    #     ref = SequenceIR(AlignLeft())
    #     ref.append(inst1)
    #     ref.append(inst2)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ref, property_set)
    #     ref = sequence_pass(ref, property_set)
    #     ref = schedule_pass(ref, property_set)
    #
    #     self.assertEqual(flat, ref)
    #
    # def test_flatten_two_levels(self):
    #     """Test that flattening works with one block and sequential instructions"""
    #     inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    #
    #     block1 = SequenceIR(AlignLeft())
    #     block1.append(inst)
    #     block = SequenceIR(AlignLeft())
    #     block.append(inst)
    #     block.append(block1)
    #
    #     ir_example = SequenceIR(AlignLeft())
    #     ir_example.append(inst)
    #     ir_example.append(block)
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     ir_example = schedule_pass(ir_example, property_set)
    #
    #     flat = ir_example.flatten()
    #     edge_list = flat.sequence.edge_list()
    #     self.assertEqual(len(edge_list), 4)
    #     self.assertTrue((0, 2) in edge_list)
    #     self.assertTrue((2, 6) in edge_list)
    #     self.assertTrue((6, 7) in edge_list)
    #     self.assertTrue((7, 1) in edge_list)
    #     self.assertEqual(flat.scheduled_elements()[0], (0, inst))
    #     self.assertEqual(flat.scheduled_elements()[1], (100, inst))
    #     self.assertEqual(flat.scheduled_elements()[2], (200, inst))
    #     # Verify that nodes removed from the graph are also removed from _time_table.
    #     self.assertEqual({x for x in flat.sequence.node_indices()}, flat._time_table.keys())

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

    # TODO : Test SequenceIR.draw()
