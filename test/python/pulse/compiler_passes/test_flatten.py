# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Schedule"""
import copy

from test import QiskitTestCase

from qiskit.pulse import (
    Constant,
    Play,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.model import QubitFrame, Qubit
from qiskit.pulse.transforms import (
    AlignLeft,
)
from qiskit.pulse.compiler import MapMixedFrame, SetSequence, SetSchedule, Flatten
from qiskit.pulse.exceptions import PulseCompilerError
from .utils import PulseIrTranspiler


class TestFlatten(QiskitTestCase):
    """Flatten tests"""

    def setUp(self):
        super().setUp()
        self._schedule_pm = PulseIrTranspiler([MapMixedFrame(), SetSequence(), SetSchedule()])
        self._flatten = Flatten()

    def _compare_scheduled_elements(self, list1, list2):
        if len(list1) != len(list2):
            return False
        for x in list1:
            if x not in list2:
                return False
        return True

    def test_flatten_ir_no_sub_blocks(self):
        """Test that flattening ir with no sub blocks doesn't do anything"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        ir_example.append(inst)
        ir_example.append(inst2)

        ir_example = self._schedule_pm.run(ir_example)
        ref = copy.deepcopy(ir_example)
        ir_example = self._flatten.run(ir_example)

        self.assertEqual(ref, ir_example)

    def test_flatten_one_sub_block(self):
        """Test that flattening works with one block"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(block)

        ir_example = self._schedule_pm.run(ir_example)
        flat = self._flatten.run(ir_example)
        edge_list = flat.sequence.edge_list()
        print(edge_list)
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 5) in edge_list)
        self.assertTrue((0, 6) in edge_list)
        self.assertTrue((5, 1) in edge_list)
        self.assertTrue((6, 1) in edge_list)
        self.assertTrue((0, inst) in flat.scheduled_elements())
        self.assertTrue((0, inst2) in flat.scheduled_elements())

    def test_flatten_one_sub_block_and_parallel_instruction(self):
        """Test that flattening works with one block and parallel instruction"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(block)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(3), target=Qubit(3)))

        ir_example = self._schedule_pm.run(ir_example)
        flat = self._flatten.run(ir_example)

        edge_list = flat.sequence.edge_list()
        self.assertEqual(len(edge_list), 6)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((0, 6) in edge_list)
        self.assertTrue((0, 7) in edge_list)
        self.assertTrue((6, 1) in edge_list)
        self.assertTrue((7, 1) in edge_list)
        self.assertTrue(
            self._compare_scheduled_elements(
                ir_example.scheduled_elements(recursive=True), flat.scheduled_elements()
            )
        )

    def test_flatten_one_sub_block_and_sequential_instructions(self):
        """Test that flattening works with one block and sequential instructions"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1)))
        ir_example.append(block)
        ir_example.append(Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2)))

        ir_example = self._schedule_pm.run(ir_example)
        flat = self._flatten.run(ir_example)

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
        self.assertTrue(
            self._compare_scheduled_elements(
                ir_example.scheduled_elements(recursive=True), flat.scheduled_elements()
            )
        )

    def test_flatten_two_sub_blocks(self):
        """Test that flattening works with two sub blocks"""
        inst1 = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(200, 0.5), frame=QubitFrame(1), target=Qubit(1))

        block1 = SequenceIR(AlignLeft())
        block1.append(inst1)
        block2 = SequenceIR(AlignLeft())
        block2.append(inst2)

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(block1)
        ir_example.append(block2)

        ir_example = self._schedule_pm.run(ir_example)
        flat = self._flatten.run(ir_example)
        ref = SequenceIR(AlignLeft())
        ref.append(inst1)
        ref.append(inst2)

        ref = self._schedule_pm.run(ref)

        self.assertEqual(flat, ref)
        self.assertTrue(
            self._compare_scheduled_elements(
                ir_example.scheduled_elements(recursive=True), flat.scheduled_elements()
            )
        )

    def test_flatten_two_levels(self):
        """Test that flattening works with one block and sequential instructions"""
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))

        block1 = SequenceIR(AlignLeft())
        block1.append(inst)
        block = SequenceIR(AlignLeft())
        block.append(inst)
        block.append(block1)

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(inst)
        ir_example.append(block)

        ir_example = self._schedule_pm.run(ir_example)
        flat = self._flatten.run(ir_example)
        edge_list = flat.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 6) in edge_list)
        self.assertTrue((6, 7) in edge_list)
        self.assertTrue((7, 1) in edge_list)
        self.assertEqual(flat.scheduled_elements()[0], (0, inst))
        self.assertEqual(flat.scheduled_elements()[1], (100, inst))
        self.assertEqual(flat.scheduled_elements()[2], (200, inst))
        # Verify that nodes removed from the graph are also removed from _time_table.
        self.assertEqual(
            {x for x in flat.sequence.node_indices() if x not in (0, 1)}, flat._time_table.keys()
        )
        self.assertTrue(
            self._compare_scheduled_elements(
                ir_example.scheduled_elements(recursive=True), flat.scheduled_elements()
            )
        )

    def test_flatten_not_scheduled(self):
        """Test that flattening an unscheduled IR raises an error"""
        ir_example = SequenceIR(AlignLeft())
        inst = Play(Constant(100, 0.5), frame=QubitFrame(1), target=Qubit(1))
        inst2 = Play(Constant(100, 0.5), frame=QubitFrame(2), target=Qubit(2))
        block = SequenceIR(AlignLeft())
        block.append(inst)
        block.append(inst2)
        ir_example.append(block)

        with self.assertRaises(PulseCompilerError):
            self._flatten.run(ir_example)
