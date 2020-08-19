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

"""Tests for drawing object of timeline drawer."""

import numpy as np

import qiskit
from qiskit.test import QiskitTestCase
from qiskit.visualization.timeline import drawing_objects, types


class TestDrawingObjects(QiskitTestCase):
    """Tests for drawing objects."""

    def setUp(self) -> None:
        """Setup."""

        # bits
        self.qubits = list(qiskit.QuantumRegister(2))

        # metadata
        self.meta1 = {'val1': 0, 'val2': 1}
        self.meta2 = {'val1': 2, 'val2': 3}

        # style data
        self.style1 = {'property1': 0, 'property2': 1}
        self.style2 = {'property1': 2, 'property2': 3}

    def test_line_data_equivalent(self):
        """Test LineData equivalent check."""
        xs = list(np.arange(10))
        ys = list(np.ones(10))

        obj1 = drawing_objects.LineData(data_type='test',
                                        bit=self.qubits[0],
                                        x=xs,
                                        y=ys,
                                        meta=self.meta1,
                                        visible=True,
                                        styles=self.style1)

        obj2 = drawing_objects.LineData(data_type='test',
                                        bit=self.qubits[0],
                                        x=xs,
                                        y=ys,
                                        meta=self.meta2,
                                        visible=False,
                                        styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_line_data_equivalent_abstract_coord(self):
        """Test LineData equivalent check with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]

        obj1 = drawing_objects.LineData(data_type='test',
                                        bit=self.qubits[0],
                                        x=xs,
                                        y=ys,
                                        meta=self.meta1,
                                        visible=True,
                                        styles=self.style1)

        obj2 = drawing_objects.LineData(data_type='test',
                                        bit=self.qubits[0],
                                        x=xs,
                                        y=ys,
                                        meta=self.meta2,
                                        visible=False,
                                        styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_box_data_equivalent(self):
        """Test BoxData equivalent check."""
        obj1 = drawing_objects.BoxData(data_type='test',
                                       bit=self.qubits[0],
                                       x0=0,
                                       y0=0,
                                       x1=1,
                                       y1=1,
                                       meta=self.meta1,
                                       visible=True,
                                       styles=self.style1)

        obj2 = drawing_objects.BoxData(data_type='test',
                                       bit=self.qubits[0],
                                       x0=0,
                                       y0=0,
                                       x1=1,
                                       y1=1,
                                       meta=self.meta2,
                                       visible=False,
                                       styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_box_data_equivalent_abstract_coord(self):
        """Test BoxData equivalent check with abstract coordinate."""
        obj1 = drawing_objects.BoxData(data_type='test',
                                       bit=self.qubits[0],
                                       x0=types.AbstractCoordinate.LEFT,
                                       y0=types.AbstractCoordinate.BOTTOM,
                                       x1=types.AbstractCoordinate.RIGHT,
                                       y1=types.AbstractCoordinate.TOP,
                                       meta=self.meta1,
                                       visible=True,
                                       styles=self.style1)

        obj2 = drawing_objects.BoxData(data_type='test',
                                       bit=self.qubits[0],
                                       x0=types.AbstractCoordinate.LEFT,
                                       y0=types.AbstractCoordinate.BOTTOM,
                                       x1=types.AbstractCoordinate.RIGHT,
                                       y1=types.AbstractCoordinate.TOP,
                                       meta=self.meta2,
                                       visible=False,
                                       styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_text_data_equivalent(self):
        """Test TextData equivalent check."""
        obj1 = drawing_objects.TextData(data_type='test',
                                        bit=self.qubits[0],
                                        x=0,
                                        y=0,
                                        text='test',
                                        latex='test',
                                        meta=self.meta1,
                                        visible=True,
                                        styles=self.style1)

        obj2 = drawing_objects.TextData(data_type='test',
                                        bit=self.qubits[0],
                                        x=0,
                                        y=0,
                                        text='test',
                                        latex='test',
                                        meta=self.meta2,
                                        visible=False,
                                        styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_text_data_equivalent_abstract_coord(self):
        """Test TextData equivalent check with abstract coordinate."""
        obj1 = drawing_objects.TextData(data_type='test',
                                        bit=self.qubits[0],
                                        x=types.AbstractCoordinate.LEFT,
                                        y=types.AbstractCoordinate.BOTTOM,
                                        text='test',
                                        latex='test',
                                        meta=self.meta1,
                                        visible=True,
                                        styles=self.style1)

        obj2 = drawing_objects.TextData(data_type='test',
                                        bit=self.qubits[0],
                                        x=types.AbstractCoordinate.LEFT,
                                        y=types.AbstractCoordinate.BOTTOM,
                                        text='test',
                                        latex='test',
                                        meta=self.meta2,
                                        visible=False,
                                        styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_bit_link_data_equivalent(self):
        """Test BitLinkData equivalent check."""
        obj1 = drawing_objects.BitLinkData(bits=[self.qubits[0], self.qubits[1]],
                                           x=0,
                                           offset=0,
                                           visible=True,
                                           styles=self.style1)

        obj2 = drawing_objects.BitLinkData(bits=[self.qubits[0], self.qubits[1]],
                                           x=0,
                                           offset=1,
                                           visible=False,
                                           styles=self.style2)

        self.assertEqual(obj1, obj2)

    def test_bit_link_data_equivalent_abstract_coord(self):
        """Test BitLinkData equivalent check with abstract coordinate."""
        obj1 = drawing_objects.BitLinkData(bits=[self.qubits[0], self.qubits[1]],
                                           x=types.AbstractCoordinate.LEFT,
                                           offset=0,
                                           visible=True,
                                           styles=self.style1)

        obj2 = drawing_objects.BitLinkData(bits=[self.qubits[0], self.qubits[1]],
                                           x=types.AbstractCoordinate.LEFT,
                                           offset=1,
                                           visible=False,
                                           styles=self.style2)

        self.assertEqual(obj1, obj2)
