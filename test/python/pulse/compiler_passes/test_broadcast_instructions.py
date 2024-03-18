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

"""Test BroadcastInstructions"""

from test import QiskitTestCase
from ddt import ddt, named_data, unpack

from qiskit.pulse import (
    Constant,
    Play,
    Delay,
    ShiftPhase,
    ShiftFrequency,
    SetFrequency,
    SetPhase,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.transforms import (
    AlignLeft,
    AlignSequential,
)
from qiskit.pulse.compiler import MapMixedFrame, SetSequence, SetSchedule, BroadcastInstructions
from qiskit.pulse.exceptions import PulseCompilerError
from .utils import PulseIrTranspiler


@ddt
class TestBroadcastInstructions(QiskitTestCase):
    """Test BroadcastInstructions"""

    def setUp(self):
        super().setUp()
        self._pm = PulseIrTranspiler([MapMixedFrame(), SetSequence(), BroadcastInstructions()])

    @named_data(
        ["set_phase", SetPhase],
        ["set_frequency", SetFrequency],
        ["shift_phase", ShiftPhase],
        ["shift_frequency", ShiftFrequency],
    )
    @unpack
    def test_all_frame_instructions(self, inst_class):
        """test frame instruction"""

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(1), QubitFrame(1))))
        ir_example.append(inst_class(1.3, frame=QubitFrame(1)))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 4)
        self.assertEqual(ir_example.sequence.num_edges(), 6)
        self.assertTrue((0, 2) in ir_example.sequence.edge_list())
        self.assertTrue((2, 3) in ir_example.sequence.edge_list())
        self.assertTrue((3, 5) in ir_example.sequence.edge_list())
        self.assertTrue((3, 6) in ir_example.sequence.edge_list())
        self.assertTrue((5, 1) in ir_example.sequence.edge_list())
        self.assertTrue((6, 1) in ir_example.sequence.edge_list())
        self.assertTrue(
            inst_class(1.3, mixed_frame=MixedFrame(Qubit(0), QubitFrame(1)))
            in ir_example.elements()
        )
        self.assertTrue(
            inst_class(1.3, mixed_frame=MixedFrame(Qubit(1), QubitFrame(1)))
            in ir_example.elements()
        )

    def test_delay_instruction(self):
        """test delay instruction"""
        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Delay(100, target=Qubit(0)))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 4)
        self.assertEqual(ir_example.sequence.num_edges(), 6)
        self.assertTrue((0, 2) in ir_example.sequence.edge_list())
        self.assertTrue((2, 3) in ir_example.sequence.edge_list())
        self.assertTrue((3, 5) in ir_example.sequence.edge_list())
        self.assertTrue((3, 6) in ir_example.sequence.edge_list())
        self.assertTrue((5, 1) in ir_example.sequence.edge_list())
        self.assertTrue((6, 1) in ir_example.sequence.edge_list())
        self.assertTrue(
            Delay(100, mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))) in ir_example.elements()
        )
        self.assertTrue(
            Delay(100, mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))) in ir_example.elements()
        )

    def test_recursion(self):
        """test that broadcasting is applied recursively"""
        sub_block = SequenceIR(AlignSequential())
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        sub_block.append(Delay(100, target=Qubit(0)))

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        sub_elements = ir_example.elements()[0].elements()
        self.assertEqual(len(sub_elements), 2)
        self.assertTrue(Delay(100, mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))) in sub_elements)

    def test_instructions_not_in_mapping(self):
        """This is an edge case where a broadcasted instruction doesn't have any corresponding
        mixed frames in the mapping. This will currently fail to sequence.
        """
        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.sequence.add_edges_from_no_data([(0, 2), (2, 1)])
        pm = PulseIrTranspiler([MapMixedFrame(), BroadcastInstructions()])

        pm.run(ir_example)
        self.assertEqual(ir_example.elements(), [Delay(100, target=Qubit(0))])

        # TODO : Once sequencing of this edge case is sorted out, ammend the test.

    def test_timing_information(self):
        """Test that timing information is carried over correctly"""
        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(Delay(100, target=Qubit(0)))

        pm = PulseIrTranspiler(
            [MapMixedFrame(), SetSequence(), SetSchedule(), BroadcastInstructions()]
        )

        ir_example = pm.run(ir_example)
        self.assertEqual(len(ir_example.time_table.keys()), 4)
        self.assertEqual(ir_example.time_table[2], 0)
        self.assertEqual(ir_example.time_table[3], 100)
        self.assertEqual(ir_example.time_table[5], 200)
        self.assertEqual(ir_example.time_table[6], 200)

    def test_multiple_successors(self):
        """Test that sequencing is done correctly with several successors"""
        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Delay(100, target=Qubit(0)))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(0))))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 4)
        edges = ir_example.sequence.edge_list()
        self.assertEqual(len(edges), 8)
        self.assertTrue((0, 5) in edges)
        self.assertTrue((0, 6) in edges)
        self.assertTrue((5, 3) in edges)
        self.assertTrue((6, 3) in edges)
        self.assertTrue((5, 4) in edges)
        self.assertTrue((6, 4) in edges)
        self.assertTrue((3, 1) in edges)
        self.assertTrue((4, 1) in edges)

    def test_no_mixed_frames_mapping(self):
        """Test that an error is raised if no mapping exists"""
        ir_example = SequenceIR(AlignLeft())

        pm = PulseIrTranspiler(BroadcastInstructions())

        with self.assertRaises(PulseCompilerError):
            pm.run(ir_example)
