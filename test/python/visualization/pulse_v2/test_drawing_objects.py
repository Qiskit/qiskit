# -*- coding: utf-8 -*-

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

# pylint: disable=missing-docstring, invalid-name

"""Tests for core modules of pulse drawer."""

import numpy as np

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import drawing_objects


class TestDrawingObjects(QiskitTestCase):
    """Tests for DrawingObjects."""
    def test_filled_area_data(self):
        """Test FilledAreaData."""
        data1 = drawing_objects.FilledAreaData(data_type='waveform',
                                               channel=pulse.DriveChannel(0),
                                               x=np.array([0, 1, 2]),
                                               y1=np.array([3, 4, 5]),
                                               y2=np.array([0, 0, 0]),
                                               meta={'test_val': 0},
                                               offset=0,
                                               scale=1,
                                               visible=True,
                                               styles={'color': 'red'})

        data2 = drawing_objects.FilledAreaData(data_type='waveform',
                                               channel=pulse.DriveChannel(0),
                                               x=np.array([0, 1, 2]),
                                               y1=np.array([3, 4, 5]),
                                               y2=np.array([0, 0, 0]),
                                               meta={'test_val': 1},
                                               offset=1,
                                               scale=2,
                                               visible=False,
                                               styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_line_data(self):
        """Test for LineData."""
        data1 = drawing_objects.LineData(data_type='baseline',
                                         channel=pulse.DriveChannel(0),
                                         x=np.array([0, 1, 2]),
                                         y=np.array([0, 0, 0]),
                                         meta={'test_val': 0},
                                         offset=0,
                                         scale=1,
                                         visible=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.LineData(data_type='baseline',
                                         channel=pulse.DriveChannel(0),
                                         x=np.array([0, 1, 2]),
                                         y=np.array([0, 0, 0]),
                                         meta={'test_val': 1},
                                         offset=1,
                                         scale=2,
                                         visible=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_text_data(self):
        """Test for TextData."""
        data1 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text',
                                         meta={'test_val': 0},
                                         offset=0,
                                         scale=1,
                                         visible=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text',
                                         meta={'test_val': 1},
                                         offset=1,
                                         scale=2,
                                         visible=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)
