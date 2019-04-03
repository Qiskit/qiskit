# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Tests for comparing the outputs of visualization tools with expected ones.
Useful for refactoring purposes."""

import os
import unittest
from codecs import encode
from math import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.tools.visualization import HAS_MATPLOTLIB, circuit_drawer
from qiskit.test import QiskitTestCase


def _path_to_diagram_reference(filename):
    return os.path.join(_this_directory(), 'references', filename)


def _this_directory():
    return os.path.dirname(os.path.abspath(__file__))


class TestVisualizationImplementation(QiskitTestCase):
    """Visual accuracy of visualization tools outputs tests."""

    latex_reference = _path_to_diagram_reference('latex_ref.png')
    matplotlib_reference = _path_to_diagram_reference('matplotlib_ref.png')
    text_reference = _path_to_diagram_reference('text_ref.txt')

    def sample_circuit(self):
        """Generate a sample circuit that includes the most common elements of
        quantum circuits.
        """
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.z(qr[0])
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])
        circuit.barrier(qr[2])
        circuit.h(qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[0])
        circuit.t(qr[0])
        circuit.tdg(qr[0])
        circuit.iden(qr[0])
        circuit.reset(qr[0])
        circuit.rx(pi, qr[0])
        circuit.ry(pi, qr[0])
        circuit.rz(pi, qr[0])
        circuit.u0(pi, qr[0])
        circuit.u1(pi, qr[0])
        circuit.u2(pi, pi, qr[0])
        circuit.u3(pi, pi, pi, qr[0])
        circuit.swap(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.ch(qr[0], qr[1])
        circuit.cu1(pi, qr[0], qr[1])
        circuit.cu3(pi, pi, pi, qr[0], qr[1])
        circuit.crz(pi, qr[0], qr[1])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.measure(qr, cr)

        return circuit

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_latex_drawer(self):
        filename = self._get_resource_path('current_latex.png')
        qc = self.sample_circuit()
        circuit_drawer(qc, filename=filename, output='latex')
        self.assertImagesAreEqual(filename, self.latex_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_matplotlib_drawer(self):
        filename = self._get_resource_path('current_matplot.png')
        qc = self.sample_circuit()
        circuit_drawer(qc, filename=filename, output='mpl')
        self.assertImagesAreEqual(filename, self.matplotlib_reference)
        os.remove(filename)

    def test_text_drawer(self):
        filename = self._get_resource_path('current_textplot.txt')
        qc = self.sample_circuit()
        output = circuit_drawer(qc, filename=filename, output="text", line_length=-1)
        self.assertFilesAreEqual(filename, self.text_reference)
        os.remove(filename)
        try:
            encode(str(output), encoding='cp437')
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should only use extended ascii (aka code page 437).")

    def assertFilesAreEqual(self, current, expected):
        """Checks if both file are the same."""
        self.assertTrue(os.path.exists(current))
        self.assertTrue(os.path.exists(expected))
        with open(current, "r", encoding='cp437') as cur, \
                open(expected, "r", encoding='cp437') as exp:
            self.assertEqual(cur.read(), exp.read())

    def assertImagesAreEqual(self, current, expected, diff_tolerance=0.001):
        """Checks if both images are similar enough to be considered equal.
        Similarity is controlled by the ```diff_tolerance``` argument."""
        from PIL import Image, ImageChops

        if isinstance(current, str):
            current = Image.open(current)
        if isinstance(expected, str):
            expected = Image.open(expected)

        diff = ImageChops.difference(expected, current)
        black_pixels = _get_black_pixels(diff)
        total_pixels = diff.size[0] * diff.size[1]
        similarity_ratio = black_pixels / total_pixels
        self.assertTrue(
            1 - similarity_ratio < diff_tolerance,
            'The images are different by more than a {}%'
            .format(diff_tolerance * 100))


def _get_black_pixels(image):
    black_and_white_version = image.convert('1')
    black_pixels = black_and_white_version.histogram()[0]
    return black_pixels


if __name__ == '__main__':
    unittest.main(verbosity=2)
