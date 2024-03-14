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
import unittest

from test import QiskitTestCase

from qiskit.pulse import (
    Constant,
    Play,
    Delay,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.transforms import (
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignEquispaced,
    AlignFunc,
)
from qiskit.pulse.compiler import MapMixedFrame, SetSequence, SetSchedule
from qiskit.pulse.exceptions import PulseCompilerError
from .utils import PulseIrTranspiler


class SchedulingTestCase(QiskitTestCase):
    """Base class for scheduling tests"""

    def setUp(self):
        super().setUp()
        self._pm = PulseIrTranspiler([MapMixedFrame(), SetSequence(), SetSchedule()])


class TestScheduleAlignLeft(SchedulingTestCase):
    """Test Schedule pass with Left Alignment"""

    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 100)

    def test_bad_or_missing_sequencing(self):
        """test that bad or missing sequencing raises an error"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)

        ir_example.sequence.add_edges_from_no_data([(1, 0), (2, 0)])
        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)

    def test_parallel_instructions(self):
        """test with two parallel instructions"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 0)

    def test_sequential_instructions(self):
        """test with two sequential instructions"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)

    def test_multiple_successors(self):
        """test for a graph where one node has several successors"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 100)

    def test_multiple_predecessors(self):
        """test for a graph where one node has several predecessors"""

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))
        ir_example.append(Delay(100, target=Qubit(0)))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 200)

    def test_recursion_to_sub_blocks(self):
        """test that scheduling is recursively applied to sub blocks which are first in the order"""

        sub_block = SequenceIR(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        sub_block = ir_example.elements()[1]
        # Note that sub blocks are oblivious to their relative timing
        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.scheduled_elements()[0][0], 0)

    def test_with_parallel_sub_block(self):
        """test with a sub block which doesn't depend on previous instructions"""

        sub_block = SequenceIR(AlignLeft())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 100)

    def test_with_sequential_sub_block_with_more_dependencies(self):
        """test with a sub block which depends on a several previous instruction"""

        sub_block = SequenceIR(AlignLeft())
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.time_table[2], 0)
        self.assertEqual(ir_example.time_table[3], 0)
        self.assertEqual(ir_example.time_table[4], 0)
        self.assertEqual(ir_example.time_table[5], 200)
        self.assertEqual(ir_example.time_table[6], 100)

    def test_sorting_effects(self):
        """test that sorting is done correctly and doesn't create side effects

        The pass relies on a topological sort of the nodes, to make sure a node is only scheduled
        after all its predecessors did. Here we test with two graphs where the ordering matters,
        to validate the sorting.

        An example problematic graph:

               2 <- 0 -> 3
               |         |
               4 <------ 5
               |
               1

        The situation will occur if node 4 is evaluated before node 5. We swap 4 and 5 next to
        have another test case.
        """

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example2 = copy.deepcopy(ir_example)

        ir_example.sequence.add_edges_from_no_data([(0, 2), (0, 3), (2, 4), (3, 5), (5, 4), (4, 1)])
        SetSchedule().run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)

        ir_example2.sequence.add_edges_from_no_data(
            [(0, 2), (0, 3), (2, 4), (3, 5), (4, 5), (5, 1)]
        )
        SetSchedule().run(ir_example2)
        self.assertEqual(ir_example2.initial_time(), 0)
        self.assertEqual(ir_example2.final_time(), 300)


class TestSetScheduleAlignRight(SchedulingTestCase):
    """Test SetSchedule with align right"""

    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 100)

    def test_bad_or_missing_sequencing(self):
        """test that bad or missing sequencing raises an error"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)

        ir_example.sequence.add_edges_from_no_data([(1, 0), (2, 0)])
        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)

    def test_parallel_instructions(self):
        """test with two parallel instructions"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.time_table[2], 100)
        self.assertEqual(ir_example.time_table[3], 0)

    def test_sequential_instructions(self):
        """test with two sequential instructions"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)

    def test_multiple_successors(self):
        """test for a graph where one node has several successors"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.scheduled_elements()[0][0], 0)
        self.assertEqual(ir_example.scheduled_elements()[1][0], 100)
        self.assertEqual(ir_example.scheduled_elements()[2][0], 200)

    def test_multiple_predecessors(self):
        """test for a graph where one node has several predecessors"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))
        ir_example.append(Delay(100, target=Qubit(0)))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.time_table[2], 100)
        self.assertEqual(ir_example.time_table[3], 0)
        self.assertEqual(ir_example.time_table[4], 200)

    def test_recursion_to_sub_blocks(self):
        """test that scheduling is recursively applied to sub blocks which are first in the order"""

        sub_block = SequenceIR(AlignRight())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        sub_block = ir_example.elements()[1]
        # Note that sub blocks are oblivious to their relative timing
        self.assertEqual(sub_block.initial_time(), 0)
        self.assertEqual(sub_block.final_time(), 100)
        self.assertEqual(sub_block.scheduled_elements()[0][0], 0)

    def test_with_parallel_sub_block(self):
        """test with a sub block which doesn't depend on previous instructions"""

        sub_block = SequenceIR(AlignRight())
        sub_block.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 200)
        self.assertEqual(ir_example._time_table[2], 100)
        self.assertEqual(ir_example._time_table[3], 0)

    def test_with_sequential_sub_block_with_more_dependencies(self):
        """test with a sub block which depends on a several previous instruction"""

        sub_block = SequenceIR(AlignRight())
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)

        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 300)
        self.assertEqual(ir_example.time_table[2], 100)
        self.assertEqual(ir_example.time_table[3], 0)
        self.assertEqual(ir_example.time_table[4], 100)
        self.assertEqual(ir_example.time_table[5], 200)
        self.assertEqual(ir_example.time_table[6], 200)


class TestScheduleAlignSequential(SchedulingTestCase):
    """Test Schedule pass with AlignSequential.

    The pass actually uses the same logic to schedule AlignSequential and AlignLeft.
    The tests are added for completeness, and in order to verify that the combination
    of SetSequence+SetSchedule works as expected.
    """

    def test_instructions_and_sub_blocks(self):
        """test with instructions and sub blocks"""

        sub_block = SequenceIR(AlignSequential())
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(200, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.initial_time(), 0)
        self.assertEqual(ir_example.final_time(), 400)
        self.assertEqual(ir_example.time_table[2], 0)
        self.assertEqual(ir_example.time_table[3], 200)
        self.assertEqual(ir_example.time_table[4], 300)

    def test_bad_or_missing_sequencing(self):
        """test that bad or missing sequencing raises an error"""

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)

        ir_example.sequence.add_edges_from_no_data([(1, 0), (2, 0)])
        with self.assertRaises(PulseCompilerError):
            SetSchedule().run(ir_example)


class TestSetScheduleAlignEquispaced(SchedulingTestCase):
    """Test SetSchedule with align equispaced"""

    @unittest.expectedFailure
    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = SequenceIR(AlignEquispaced(100))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        self._pm.run(ir_example)

    # TODO : Implement align equispaced.


class TestSetScheduleAlignFunc(SchedulingTestCase):
    """Test SetSchedule with align equispaced"""

    @unittest.expectedFailure
    def test_single_instruction(self):
        """test with a single instruction"""

        ir_example = SequenceIR(AlignFunc(100, lambda x: x))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        self._pm.run(ir_example)

    # TODO : Implement align func.
