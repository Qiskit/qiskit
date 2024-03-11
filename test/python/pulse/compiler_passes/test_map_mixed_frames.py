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

"""Test MapMixedFrames"""
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

from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.transforms import AlignLeft
from qiskit.pulse.compiler import MapMixedFrame


class TestMapMixedFrames(QiskitTestCase):
    """Test MapMixedFrames analysis pass"""

    def test_basic_ir(self):
        """test with basic IR"""
        ir_example = SequenceIR(AlignLeft())
        mf = MixedFrame(Qubit(0), QubitFrame(1))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf))

        mapping_pass = MapMixedFrame()
        mapping_pass.run(ir_example)
        self.assertEqual(len(mapping_pass.property_set.keys()), 1)
        mapping = mapping_pass.property_set["mixed_frames_mapping"]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[mf.pulse_target], {mf})
        self.assertEqual(mapping[mf.frame], {mf})

        mf2 = MixedFrame(Qubit(0), QubitFrame(2))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf2))

        mapping_pass.run(ir_example)
        mapping = mapping_pass.property_set["mixed_frames_mapping"]
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[mf.pulse_target], {mf, mf2})
        self.assertEqual(mapping[mf.frame], {mf})

    def test_with_several_inst_target_types(self):
        """test with different inst_target types"""
        ir_example = SequenceIR(AlignLeft())
        mf = MixedFrame(Qubit(0), QubitFrame(1))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf))
        ir_example.append(Delay(100, target=Qubit(2)))
        ir_example.append(ShiftPhase(100, frame=QubitFrame(2)))

        mapping_pass = MapMixedFrame()
        mapping_pass.run(ir_example)
        mapping = mapping_pass.property_set["mixed_frames_mapping"]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[Qubit(0)], {mf})
        self.assertEqual(mapping[QubitFrame(1)], {mf})

    def test_with_sub_blocks(self):
        """test with sub blocks"""
        mf1 = MixedFrame(Qubit(0), QubitFrame(0))
        mf2 = MixedFrame(Qubit(0), QubitFrame(1))
        mf3 = MixedFrame(Qubit(0), QubitFrame(2))

        sub_block_2 = SequenceIR(AlignLeft())
        sub_block_2.append(Play(Constant(100, 0.1), mixed_frame=mf1))

        sub_block_1 = SequenceIR(AlignLeft())
        sub_block_1.append(Play(Constant(100, 0.1), mixed_frame=mf2))
        sub_block_1.append(sub_block_2)

        mapping_pass = MapMixedFrame()
        mapping_pass.run(sub_block_1)
        mapping = mapping_pass.property_set["mixed_frames_mapping"]

        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[Qubit(0)], {mf1, mf2})
        self.assertEqual(mapping[QubitFrame(0)], {mf1})
        self.assertEqual(mapping[QubitFrame(1)], {mf2})

        ir_example = SequenceIR(AlignLeft())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=mf3))
        ir_example.append(sub_block_1)

        mapping_pass = MapMixedFrame()
        mapping_pass.run(ir_example)
        mapping = mapping_pass.property_set["mixed_frames_mapping"]

        self.assertEqual(len(mapping), 4)
        self.assertEqual(mapping[Qubit(0)], {mf1, mf2, mf3})
        self.assertEqual(mapping[QubitFrame(0)], {mf1})
        self.assertEqual(mapping[QubitFrame(1)], {mf2})
        self.assertEqual(mapping[QubitFrame(2)], {mf3})

    def test_equating(self):
        """Test equating of passes"""
        self.assertTrue(MapMixedFrame() == MapMixedFrame())
        self.assertFalse(MapMixedFrame() == QubitFrame(1))
