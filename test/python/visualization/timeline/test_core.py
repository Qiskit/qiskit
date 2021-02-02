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

"""Tests for core modules of timeline drawer."""

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.visualization.timeline import core, stylesheet, generators, layouts


class TestCanvas(QiskitTestCase):
    """Test for canvas."""

    def setUp(self):
        super().setUp()

        self.style = stylesheet.QiskitTimelineStyle()
        circ = QuantumCircuit(4)
        circ.h(0)
        circ.barrier()
        circ.cx(0, 2)
        circ.cx(1, 3)

        self.circ = transpile(circ,
                              scheduling_method='alap',
                              basis_gates=['h', 'cx'],
                              instruction_durations=[('h', 0, 200),
                                                     ('cx', [0, 2], 1000),
                                                     ('cx', [1, 3], 1000)],
                              optimization_level=0)

    def test_time_range(self):
        """Test calculating time range."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter = {
            'margin.left_percent': 0.1,
            'margin.right_percent': 0.1
        }
        canvas.time_range = (0, 100)

        ref_range = [-10., 110.]
        self.assertListEqual(list(canvas.time_range), ref_range)

    def test_load_program(self):
        """Test loading program."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            'gates': [generators.gen_sched_gate],
            'bits': [],
            'barriers': [],
            'gate_links': []
        }
        canvas.layout = {
            'bit_arrange': layouts.qreg_creg_ascending,
            'time_axis_map': layouts.time_map_in_dt
        }

        canvas.load_program(self.circ)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 8)

        ref_coord = {
            self.circ.qregs[0][0]: -1.,
            self.circ.qregs[0][1]: -2.,
            self.circ.qregs[0][2]: -3.,
            self.circ.qregs[0][3]: -4.
        }
        self.assertDictEqual(canvas.assigned_coordinates, ref_coord)

    def test_gate_link_overlap(self):
        """Test shifting gate link overlap."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter.update({
            'formatter.margin.link_interval_dt': 20
        })
        canvas.generator = {
            'gates': [],
            'bits': [],
            'barriers': [],
            'gate_links': [generators.gen_gate_link]
        }
        canvas.layout = {
            'bit_arrange': layouts.qreg_creg_ascending,
            'time_axis_map': layouts.time_map_in_dt
        }

        canvas.load_program(self.circ)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 2)

        self.assertListEqual(drawings_tested[0][1].xvals, [710.])
        self.assertListEqual(drawings_tested[1][1].xvals, [690.])

        ref_keys = list(canvas._collections.keys())
        self.assertEqual(drawings_tested[0][0], ref_keys[0])
        self.assertEqual(drawings_tested[1][0], ref_keys[1])

    def test_object_outside_xlimit(self):
        """Test eliminating drawings outside the horizontal limit."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            'gates': [generators.gen_sched_gate],
            'bits': [generators.gen_bit_name, generators.gen_timeslot],
            'barriers': [],
            'gate_links': []
        }
        canvas.layout = {
            'bit_arrange': layouts.qreg_creg_ascending,
            'time_axis_map': layouts.time_map_in_dt
        }

        canvas.load_program(self.circ)
        canvas.set_time_range(t_start=400, t_end=600)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 12)

    def test_non_transpiled_delay_circuit(self):
        """Test non-transpiled circuit containing instruction which is trivial on duration."""
        circ = QuantumCircuit(1)
        circ.delay(10, 0)

        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            'gates': [generators.gen_sched_gate],
            'bits': [],
            'barriers': [],
            'gate_links': []
        }

        canvas.load_program(circ)
        self.assertEqual(len(canvas._collections), 1)
