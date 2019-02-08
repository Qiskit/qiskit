# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for circuit_drawer."""
import os
from PIL import ImageChops

from qiskit.test import QiskitTestCase


def _get_black_pixels(image):
    black_and_white_version = image.convert('1')
    black_pixels = black_and_white_version.histogram()[0]
    return black_pixels


class DrawingTestCase(QiskitTestCase):
    """This class intends to streamline the class hierarchy and to avoid code repetition.
    In particular, it combines two assertions which are used by TestDrawingMethods and
    TestVisualizationImplementation.

    """
    def assertFilesAreEqual(self, current, expected):
        """Checks if both files are the same.

        Args:
            current (str): path to the first of compared files
            expected (str): path to the second of compared files

        Returns:
            bool: True, if files coincide bytewise, False otherwise.

        """
        self.assertTrue(os.path.exists(current))
        self.assertTrue(os.path.exists(expected))

        with open(current, "r", encoding='cp437') as cur, \
                open(expected, "r", encoding='cp437') as exp:
            self.assertEqual(cur.read(), exp.read(), msg='{} file differs from {}'.format(current,
                                                                                          expected))

    def assertImagesAreEqual(self, current, expected, diff_tolerance=0.001):
        """
        Checks if both images are similar enough to be considered equal.
        Similarity is controlled with the ```diff_tolerance``` argument.

        Args:
            current (str): path to the first of compared images
            expected (str): path to the second of compared images
            diff_tolerance (float): a maximum acceptable difference between the compared images

        Returns:
            bool: True, if images coincide up to given precision, False otherwise.
        """
        from PIL import Image as im

        current_im = im.open(current)
        expected_im = im.open(expected)

        diff = ImageChops.difference(expected_im, current_im)
        black_pixels = _get_black_pixels(diff)
        total_pixels = diff.size[0] * diff.size[1]
        similarity_ratio = black_pixels / total_pixels
        self.assertTrue(
            1 - similarity_ratio < diff_tolerance,
            msg='The image {} differs from {} by more than a {}%'
            .format(current, expected, diff_tolerance * 100))
