# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for generator of timeline drawer."""

import qiskit
from qiskit.test import QiskitTestCase
from qiskit.visualization.timeline import generators, types, stylesheet
from qiskit.circuit import library



class TestGates(QiskitTestCase):
    """Tests for generator.gates."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qubit = list(qiskit.QuantumRegister(1))[0]

        self.u1 = types.ScheduledGate(
            t0=100, operand=library.U1Gate(0),
            duration=0, bits=[self.qubit], bit_position=0)

        self.u3 = types.ScheduledGate(
            t0=100, operand=library.U3Gate(0, 0, 0),
            duration=20, bits=[self.qubit], bit_position=0)

        style = stylesheet.QiskitTimelineStyle()
        self.formatter = style.formatter

    def test_gen_sched_gate_with_finite_duration(self):
        """Test test_gen_sched_gate with finite duration gate."""
        drawing_obj = generators.gen_sched_gate(self.u3, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, types.DrawingBox.SCHED_GATE)
        self.assertListEqual(list(drawing_obj.xvals), [100, 120])
        self.assertListEqual(list(drawing_obj.yvals), [-0.5 * self.formatter['box_height.gate'],
                                                       0.5 * self.formatter['box_height.gate']])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        ref_meta = {
            'name': 'u3',
            'label': 'n/a',
            'bits': 'q0',
            't0': 100,
            'duration': 20,
            'unitary': '[[1.+0.j 0.-0.j]\n [0.+0.j 1.+0.j]]',
            'parameters': '0, 0, 0'
        }
        self.assertDictEqual(ref_meta, drawing_obj.meta)

        ref_styles = {
            'zorder': self.formatter['layer.gate'],
            'facecolor': self.formatter['gate_face_color.u3'],
            'alpha': self.formatter['alpha.gate'],
            'linewidth': self.formatter['line_width.gate']
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

