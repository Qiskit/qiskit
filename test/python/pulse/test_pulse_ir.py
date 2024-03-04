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
    MemorySlot,
    Play,
    Delay,
    Acquire,
    ShiftPhase,
    DriveChannel,
    AcquireChannel,
)

from qiskit.pulse.ir import (
    IrBlock,
)

from qiskit.pulse.ir.alignments import AlignLeft
from qiskit.pulse.model import QubitFrame, Qubit
from qiskit.pulse.compiler.temp_passes import (
    sequence_pass,
    schedule_pass,
    analyze_target_frame_pass,
)


class TestIrBlock(QiskitTestCase):
    """Test IrBlock objects"""

    _delay_inst = Delay(50, channel=DriveChannel(0))
    _play_inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
    _shift_phase_inst = ShiftPhase(0.1, channel=DriveChannel(3))
    _acquire_inst = Acquire(200, channel=AcquireChannel(2), mem_slot=MemorySlot(4))

    def test_ir_creation(self):
        """Test ir creation"""
        ir_example = IrBlock(AlignLeft())
        self.assertEqual(ir_example.sequence.num_nodes(), 2)
        self.assertEqual(ir_example.initial_time(), None)
        self.assertEqual(ir_example.final_time(), None)
        self.assertEqual(ir_example.duration, None)

    def test_add_elements(self):
        """Test addition of elements"""
        ir_example = IrBlock(AlignLeft())
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
        ir_example = IrBlock(AlignLeft())
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

    def test_final_time(self):
        """Test final time"""
        ir_example = IrBlock(AlignLeft())
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

    def test_duration(self):
        """Test duration"""
        ir_example = IrBlock(AlignLeft())
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
        sub_block = IrBlock(AlignLeft())
        sub_block.append(inst)
        sub_block._time_table[2] = 0
        sub_block._sequence.add_edge(0, 2, None)
        sub_block._sequence.add_edge(2, 1, None)

        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.duration, 100)

        ir_example = IrBlock(AlignLeft())
        ir_example.append(inst)
        ir_example.append(sub_block)
        ir_example._time_table[2] = 100
        ir_example._time_table[3] = 300
        ir_example._sequence.add_edge(0, 2, None)
        ir_example._sequence.add_edge(3, 1, None)

        self.assertEqual(ir_example.initial_time(), 100)
        self.assertEqual(ir_example.final_time(), 400)
        self.assertEqual(ir_example.duration, 300)

    def test_flatten_ir_no_sub_blocks(self):
        """Test that flattening ir with no sub blocks doesn't do anything"""
        ir_example = IrBlock(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        ir_example.append(inst)
        ir_example.append(inst2)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        self.assertEqual(ir_example.flatten(), ir_example)

    def test_flatten_inplace_flag(self):
        """Test that inplace flag in flattening works"""
        ir_example = IrBlock(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        ir_example.append(inst)
        ir_example.append(inst2)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        self.assertTrue(ir_example.flatten(inplace=True) is ir_example)
        self.assertFalse(ir_example.flatten() is ir_example)

    def test_flatten_one_sub_block(self):
        """Test that flattening works with one block"""
        ir_example = IrBlock(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = IrBlock(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        flat = ir_example.flatten()
        edge_list = flat.sequence.edge_list()
        print(edge_list)
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 5) in edge_list)
        self.assertTrue((0, 6) in edge_list)
        self.assertTrue((5, 1) in edge_list)
        self.assertTrue((6, 1) in edge_list)

    def test_flatten_one_sub_block_and_parallel_instruction(self):
        """Test that flattening works with one block and parallel instruction"""
        ir_example = IrBlock(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = IrBlock(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(block)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(3), target=Qubit(3)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        flat = ir_example.flatten()
        edge_list = flat.sequence.edge_list()
        self.assertEqual(len(edge_list), 6)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((0, 6) in edge_list)
        self.assertTrue((0, 7) in edge_list)
        self.assertTrue((6, 1) in edge_list)
        self.assertTrue((7, 1) in edge_list)

    def test_flatten_one_sub_block_and_sequential_instructions(self):
        """Test that flattening works with one block and sequential instructions"""
        ir_example = IrBlock(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = IrBlock(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1)))
        ir_example.append(block)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2)))

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        flat = ir_example.flatten()
        edge_list = flat.sequence.edge_list()
        self.assertEqual(len(edge_list), 8)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((4, 1) in edge_list)
        self.assertTrue((2, 7) in edge_list)
        self.assertTrue((2, 8) in edge_list)
        self.assertTrue((7, 1) in edge_list)
        self.assertTrue((8, 1) in edge_list)
        self.assertTrue((7, 4) in edge_list)
        self.assertTrue((8, 4) in edge_list)

    def test_flatten_two_levels(self):
        """Test that flattening works with one block and sequential instructions"""
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))

        block1 = IrBlock(AlignLeft())
        block1.append(inst)
        block = IrBlock(AlignLeft())
        block.append(inst)
        block.append(block1)

        ir_example = IrBlock(AlignLeft())
        ir_example.append(inst)
        ir_example.append(block)

        property_set = {}
        analyze_target_frame_pass(ir_example, property_set)
        ir_example = sequence_pass(ir_example, property_set)
        ir_example = schedule_pass(ir_example, property_set)

        flat = ir_example.flatten()
        edge_list = flat.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 6) in edge_list)
        self.assertTrue((6, 7) in edge_list)
        self.assertTrue((7, 1) in edge_list)
        self.assertEqual(flat.scheduled_elements()[0], [0, inst])
        self.assertEqual(flat.scheduled_elements()[1], [100, inst])
        self.assertEqual(flat.scheduled_elements()[2], [200, inst])

    # TODO : Test IrBlock equating. Problem with Alignment, and possibly InNode,OutNode.

    # TODO : Test IrBlock.draw()
