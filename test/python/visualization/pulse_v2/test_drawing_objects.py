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
                                               fix_position=True,
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
                                               fix_position=False,
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
                                         fix_position=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.LineData(data_type='baseline',
                                         channel=pulse.DriveChannel(0),
                                         x=np.array([0, 1, 2]),
                                         y=np.array([0, 0, 0]),
                                         meta={'test_val': 1},
                                         offset=1,
                                         scale=2,
                                         visible=False,
                                         fix_position=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_text_data(self):
        """Test for TextData."""
        data1 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text1',
                                         latex='my_syntax1',
                                         meta={'test_val': 0},
                                         offset=0,
                                         scale=1,
                                         visible=True,
                                         fix_position=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text2',
                                         latex='my_syntax2',
                                         meta={'test_val': 1},
                                         offset=1,
                                         scale=2,
                                         visible=False,
                                         fix_position=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_filled_area_data_compression(self):
        """Test for ndarray compression with filled area data."""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        y1 = np.array([0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6])
        y2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        data = drawing_objects.FilledAreaData(data_type='data',
                                              channel=pulse.DriveChannel(0),
                                              x=x,
                                              y1=y1,
                                              y2=y2)

        ref_x = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13])
        ref_y1 = np.array([0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6])
        ref_y2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        np.testing.assert_array_almost_equal(data.x, ref_x)
        np.testing.assert_array_almost_equal(data.y1, ref_y1)
        np.testing.assert_array_almost_equal(data.y2, ref_y2)

    def test_line_data_compression(self):
        """Test for ndarray compression with line data."""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        y = np.array([0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6])

        data = drawing_objects.LineData(data_type='data',
                                        channel=pulse.DriveChannel(0),
                                        x=x,
                                        y=y)

        ref_x = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13])
        ref_y = np.array([0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6])

        np.testing.assert_array_almost_equal(data.x, ref_x)
        np.testing.assert_array_almost_equal(data.y, ref_y)
