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

"""Test compiler passes"""

from test import QiskitTestCase
from qiskit.pulse import (
    Constant,
    Play,
    Delay,
    ShiftPhase,
)

from qiskit.pulse.ir import (
    IrBlock,
)

from qiskit.pulse.ir.alignments import AlignLeft
from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.compiler import analyze_target_frame_pass, sequence_pass, schedule_pass


class TestAnalyzeTargetFramePass(QiskitTestCase):
    """Test analyze_target_frame_pass"""

    def test_basic_ir(self):
        """test with basic IR"""
        ir_example = IrBlock(AlignLeft())
        mf = MixedFrame(Qubit(0), QubitFrame(1))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        self.assertEqual(len(property_set.keys()), 1)
        mapping = property_set["target_frame_map"]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[mf.pulse_target], {mf})
        self.assertEqual(mapping[mf.frame], {mf})

        mf2 = MixedFrame(Qubit(0), QubitFrame(2))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf2))
        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        mapping = property_set["target_frame_map"]
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[mf.pulse_target], {mf, mf2})
        self.assertEqual(mapping[mf.frame], {mf})

    def test_with_several_inst_target_types(self):
        """test with different inst_target types"""
        ir_example = IrBlock(AlignLeft())
        mf = MixedFrame(Qubit(0), QubitFrame(1))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf))
        ir_example.append(Delay(100, target=Qubit(2)))
        ir_example.append(ShiftPhase(100, frame=QubitFrame(2)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        mapping = property_set["target_frame_map"]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[Qubit(0)], {mf})
        self.assertEqual(mapping[QubitFrame(1)], {mf})

    def test_with_sub_blocks(self):
        """test with sub blocks"""
        mf1 = MixedFrame(Qubit(0), QubitFrame(0))
        mf2 = MixedFrame(Qubit(0), QubitFrame(1))
        mf3 = MixedFrame(Qubit(0), QubitFrame(2))

        sub_block_2 = IrBlock(AlignLeft())
        sub_block_2.append(Play(Constant(100, 0.1), mixed_frame=mf1))

        sub_block_1 = IrBlock(AlignLeft())
        sub_block_1.append(Play(Constant(100, 0.1), mixed_frame=mf2))
        sub_block_1.append(sub_block_2)

        property_set = {}
        analyze_target_frame_pass(sub_block_1, property_set)
        mapping = property_set["target_frame_map"]
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[Qubit(0)], {mf1, mf2})
        self.assertEqual(mapping[QubitFrame(0)], {mf1})
        self.assertEqual(mapping[QubitFrame(1)], {mf2})

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf3))
        ir_example.append(sub_block_1)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        mapping = property_set["target_frame_map"]
        self.assertEqual(len(mapping), 4)
        self.assertEqual(mapping[Qubit(0)], {mf1, mf2, mf3})
        self.assertEqual(mapping[QubitFrame(0)], {mf1})
        self.assertEqual(mapping[QubitFrame(1)], {mf2})
        self.assertEqual(mapping[QubitFrame(2)], {mf3})


class TestSequencePassAlignLeft(QiskitTestCase):
    """Test sequence_pass with align left"""

    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 2)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 1) in edge_list)

    # TODO: Take care of this weird edge case
    # def test_instruction_not_in_mapping(self):
    #     """test with an instruction which is not in the mapping"""
    #
    #     ir_example = IrBlock(AlignLeft())
    #     ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
    #     ir_example.append(Delay(100, target=Qubit(5)))
    #
    #     property_set = {}
    #     analyze_target_frame_pass(ir_example, property_set)
    #     ir_example = sequence_pass(ir_example, property_set)
    #     edge_list = ir_example.sequence.edge_list()
    #     self.assertEqual(len(edge_list), 4)
    #     self.assertTrue((0, 2) in edge_list)
    #     self.assertTrue((0, 3) in edge_list)
    #     self.assertTrue((2, 1) in edge_list)
    #     self.assertTrue((3, 1) in edge_list)

    def test_parallel_instructions(self):
        """test with two parallel instructions"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    def test_sequential_instructions(self):
        """test with two sequential instructions"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 3)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    def test_pulse_target_instruction_broadcasting_to_children(self):
        """test with an instruction which is defined on a PulseTarget and is
        broadcasted to several children"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_frame_instruction_broadcasting_to_children(self):
        """test with an instruction which is defined on a Frame and is broadcasted to several children"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(ShiftPhase(100, frame=QubitFrame(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(0))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_pulse_target_instruction_dependency(self):
        """test with an instruction which is defined on a PulseTarget and depends on
        several mixed frames"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))
        ir_example.append(Delay(100, target=Qubit(0)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_frame_instruction_dependency(self):
        """test with an instruction which is defined on a Frame and depends on several mixed frames"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(0))))
        ir_example.append(ShiftPhase(100, frame=QubitFrame(0)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_recursion_to_sub_blocks(self):
        """test that sequencing is recursively applied to sub blocks"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)

        edge_list_sub_block = ir_example.elements()[1].sequence.edge_list()
        self.assertEqual(len(edge_list_sub_block), 2)
        self.assertTrue((0, 2) in edge_list_sub_block)
        self.assertTrue((2, 1) in edge_list_sub_block)

    def test_with_parallel_sub_block(self):
        """test with a sub block which doesn't depend on previous instructions"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    def test_with_simple_sequential_sub_block(self):
        """test with a sub block which depends on a single previous instruction"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_with_sequential_sub_block_with_more_dependencies(self):
        """test with a sub block which depends on a single previous instruction"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 8)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((0, 4) in edge_list)
        self.assertTrue((2, 5) in edge_list)
        self.assertTrue((3, 5) in edge_list)
        self.assertTrue((4, 6) in edge_list)
        self.assertTrue((5, 1) in edge_list)
        self.assertTrue((6, 1) in edge_list)


class TestSchedulePassAlignLeft(QiskitTestCase):
    """Test schedule_pass with align left"""

    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 100)

    def test_parallel_instructions(self):
        """test with two parallel instructions"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 0)

    def test_sequential_instructions(self):
        """test with two sequential instructions"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)

    def test_multiple_children(self):
        """test for a graph where one node has several children"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 100)

    def test_multiple_parents(self):
        """test for a graph where one node has several parents"""

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))
        ir_example.append(Delay(100, target=Qubit(0)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 200)

    def test_recursion_to_leading_sub_blocks(self):
        """test that scheduling is recursively applied to sub blocks which are first in the order"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        sub_block = ir_example.elements()[1]
        # Note that sub blocks are oblivious to their relative timing
        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.scheduled_elements()[0][0], 0)

    def test_recursion_to_non_leading_sub_blocks(self):
        """test that scheduling is recursively applied to sub blocks when they are not first in order"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        sub_block = ir_example.elements()[2]
        # Note that sub blocks are oblivious to their relative timing
        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)

    def test_with_parallel_sub_block(self):
        """test with a sub block which doesn't depend on previous instructions"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 100)

    def test_with_sequential_sub_block_with_more_dependencies(self):
        """test with a sub block which depends on a single previous instruction"""

        sub_block = IrBlock(AlignLeft())
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = IrBlock(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[3][0], 200)
        self.assertEqual(ir_example.scheduled_elements()[4][0], 100)
