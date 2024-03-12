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

"""Test SetSequence"""
import unittest
from test import QiskitTestCase
from ddt import ddt, named_data, unpack

from qiskit.pulse import (
    Constant,
    Play,
    Delay,
    ShiftPhase,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.transforms import (
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignFunc,
    AlignEquispaced,
)
from qiskit.pulse.compiler import MapMixedFrame, SetSequence
from qiskit.pulse.exceptions import PulseCompilerError
from .utils import PulseIrTranspiler


class TestSetSequence(QiskitTestCase):
    """General tests for set sequence pass"""

    def test_equating(self):
        """Test pass equating"""
        self.assertTrue(SetSequence() == SetSequence())
        self.assertFalse(SetSequence() == MapMixedFrame())


@ddt
class TestSetSequenceParallelAlignment(QiskitTestCase):
    """Test SetSequence pass with Parallel Alignment"""

    ddt_named_data = [["align_left", AlignLeft()], ["align_right", AlignRight()]]

    def setUp(self):
        super().setUp()
        self._pm = PulseIrTranspiler([MapMixedFrame(), SetSequence()])

    @named_data(*ddt_named_data)
    @unpack
    def test_no_mapping_pass_error(self, alignment):
        """test that running without MapMixedFrame pass raises a PulseError"""

        pm = PulseIrTranspiler()
        pm.append(SetSequence())
        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        with self.assertRaises(PulseCompilerError):
            pm.run(ir_example)

    # TODO: Take care of this weird edge case
    @unittest.expectedFailure
    @named_data(*ddt_named_data)
    @unpack
    def test_instruction_not_in_mapping(self, alignment):
        """test with an instruction which is not in the mapping"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Delay(100, target=Qubit(5)))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_parallel_instructions(self, alignment):
        """test with two parallel instructions"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_sequential_instructions(self, alignment):
        """test with two sequential instructions"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 3)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_pulse_target_instruction_sequencing_to_dependent_instructions(self, alignment):
        """test that an instruction which is defined on a PulseTarget and is sequenced correctly
        to several dependent isntructions"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_frame_instruction_broadcasting_to_dependent_instructions(self, alignment):
        """test that an instruction which is defined on a Frame is correctly sequenced to several
        dependent instructions"""

        ir_example = SequenceIR(alignment)
        ir_example.append(ShiftPhase(100, frame=QubitFrame(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 7)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)
        self.assertTrue((0, 5) in edge_list)
        self.assertTrue((5, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_pulse_target_instruction_dependency(self, alignment):
        """test with an instruction which is defined on a PulseTarget and depends on
        several mixed frames"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))
        ir_example.append(Delay(100, target=Qubit(0)))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_frame_instruction_dependency(self, alignment):
        """test with an instruction which is defined on a Frame and depends on several mixed frames"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(0))))
        ir_example.append(ShiftPhase(100, frame=QubitFrame(0)))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 4) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_recursion_to_sub_blocks(self, alignment):
        """test that sequencing is recursively applied to sub blocks"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        edge_list_sub_block = ir_example.elements()[1].sequence.edge_list()
        self.assertEqual(len(edge_list_sub_block), 2)
        self.assertTrue((0, 2) in edge_list_sub_block)
        self.assertTrue((2, 1) in edge_list_sub_block)

    @named_data(*ddt_named_data)
    @unpack
    def test_with_parallel_sub_block(self, alignment):
        """test with a sub block which doesn't depend on previous instructions"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_with_simple_sequential_sub_block(self, alignment):
        """test with a sub block which depends on a single previous instruction"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_with_sequential_sub_block_with_more_dependencies(self, alignment):
        """test with a sub block which depends on several previous instruction"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
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


@ddt
class TestSetSequenceSequentialAlignment(QiskitTestCase):
    """Test SetSequence pass with Sequential Alignment"""

    ddt_named_data = [
        ["align_sequential", AlignSequential()],
        ["align_func", AlignFunc(100, lambda x: x)],
        ["align_equispaced", AlignEquispaced(100)],
    ]

    def setUp(self):
        super().setUp()
        self._pm = PulseIrTranspiler([MapMixedFrame(), SetSequence()])

    @named_data(*ddt_named_data)
    @unpack
    def test_no_mapping_pass_error(self, alignment):
        """test that running without MapMixedFrame pass raises a PulseError"""

        pm = PulseIrTranspiler()
        pm.append(SetSequence())
        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        with self.assertRaises(PulseCompilerError):
            pm.run(ir_example)

    @named_data(*ddt_named_data)
    @unpack
    def test_several_instructions(self, alignment):
        """test with several instructions"""

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(2), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    @named_data(*ddt_named_data)
    @unpack
    def test_recursion_to_sub_blocks(self, alignment):
        """test that sequencing is recursively applied to sub blocks"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)

        edge_list_sub_block = ir_example.elements()[1].sequence.edge_list()
        self.assertEqual(len(edge_list_sub_block), 2)
        self.assertTrue((0, 2) in edge_list_sub_block)
        self.assertTrue((2, 1) in edge_list_sub_block)

    @named_data(*ddt_named_data)
    @unpack
    def test_sub_blocks_and_instructions(self, alignment):
        """test sequencing with a mix of instructions and sub blocks"""

        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = SequenceIR(alignment)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(2))))

        ir_example = self._pm.run(ir_example)

        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 3) in edge_list)
        self.assertTrue((3, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)
