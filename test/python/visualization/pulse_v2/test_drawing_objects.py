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

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import drawing_objects, types


class TestDrawingObjects(QiskitTestCase):
    """Tests for DrawingObjects."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        # bits
        self.ch_d0 = [pulse.DriveChannel(0)]
        self.ch_d1 = [pulse.DriveChannel(1)]

        # metadata
        self.meta1 = {'val1': 0, 'val2': 1}
        self.meta2 = {'val1': 2, 'val2': 3}

        # style data
        self.style1 = {'property1': 0, 'property2': 1}
        self.style2 = {'property1': 2, 'property2': 3}

    def test_line_data_equivalent(self):
        """Test for LineData."""
        xs = [0, 1, 2]
        ys = [3, 4, 5]

        data1 = drawing_objects.LineData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         channels=self.ch_d0,
                                         meta=self.meta1,
                                         ignore_scaling=True,
                                         styles=self.style1)

        data2 = drawing_objects.LineData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         channels=self.ch_d1,
                                         meta=self.meta2,
                                         ignore_scaling=True,
                                         styles=self.style2)

        self.assertEqual(data1, data2)

    def test_line_data_equivalent_with_abstract_coordinate(self):
        """Test for LineData with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.TOP, types.AbstractCoordinate.BOTTOM]

        data1 = drawing_objects.LineData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         channels=self.ch_d0,
                                         meta=self.meta1,
                                         ignore_scaling=True,
                                         styles=self.style1)

        data2 = drawing_objects.LineData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         channels=self.ch_d1,
                                         meta=self.meta2,
                                         ignore_scaling=True,
                                         styles=self.style2)

        self.assertEqual(data1, data2)

    def test_text_data(self):
        """Test for TextData."""
        xs = [0]
        ys = [1]

        data1 = drawing_objects.TextData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         text='test1',
                                         latex=r'test_1',
                                         channels=self.ch_d0,
                                         meta=self.meta1,
                                         ignore_scaling=True,
                                         styles=self.style1)

        data2 = drawing_objects.TextData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         text='test2',
                                         latex=r'test_2',
                                         channels=self.ch_d1,
                                         meta=self.meta2,
                                         ignore_scaling=True,
                                         styles=self.style2)

        self.assertEqual(data1, data2)

    def test_text_data_with_abstract_coordinate(self):
        """Test for TextData with abstract coordinate."""
        xs = [types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.TOP]

        data1 = drawing_objects.TextData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         text='test1',
                                         latex=r'test_1',
                                         channels=self.ch_d0,
                                         meta=self.meta1,
                                         ignore_scaling=True,
                                         styles=self.style1)

        data2 = drawing_objects.TextData(data_type='test',
                                         xvals=xs,
                                         yvals=ys,
                                         text='test2',
                                         latex=r'test_2',
                                         channels=self.ch_d1,
                                         meta=self.meta2,
                                         ignore_scaling=True,
                                         styles=self.style2)

        self.assertEqual(data1, data2)
