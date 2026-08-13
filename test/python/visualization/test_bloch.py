# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit.visualization.plot_bloch_vector and plot_bloch_multivector"""

import math
import unittest
from io import BytesIO

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.utils import optionals
from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector

from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase

if optionals.HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
if optionals.HAS_PIL:
    from PIL import Image


@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
@unittest.skipUnless(optionals.HAS_PIL, "PIL not available")
class TestPlotBlochVector(QiskitVisualizationTestCase):
    """Visual tests for plot_bloch_vector."""

    def test_plot_bloch_vector_cartesian(self):
        """A Cartesian Bloch vector renders the expected sphere."""
        img_ref = path_to_diagram_reference("bloch_vector_cartesian.png")
        fig = plot_bloch_vector([0, 1, 0], title="New Bloch Sphere")
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    def test_plot_bloch_vector_spherical(self):
        """A spherical-coordinate Bloch vector renders the same sphere as its
        Cartesian equivalent."""
        img_ref = path_to_diagram_reference("bloch_vector_spherical.png")
        fig = plot_bloch_vector(
            [1, math.pi / 2, math.pi / 3], coord_type="spherical", title="New Bloch Sphere"
        )
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    def test_plot_bloch_vector_external_axes(self):
        """Rendering onto caller-supplied axes mutates them in place and
        returns ``None``."""
        img_ref = path_to_diagram_reference("bloch_vector_external_axes.png")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        result = plot_bloch_vector([0, 0, 1], ax=ax)
        self.assertIsNone(result)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)


@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
@unittest.skipUnless(optionals.HAS_PIL, "PIL not available")
class TestPlotBlochMultivector(QiskitVisualizationTestCase):
    """Visual tests for plot_bloch_multivector."""

    def setUp(self):
        super().setUp()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        self.state = Statevector(qc)

    def test_plot_bloch_multivector(self):
        """A multi-qubit state renders one Bloch sphere per qubit, in
        qubit order."""
        img_ref = path_to_diagram_reference("bloch_multivector.png")
        fig = plot_bloch_multivector(self.state, title="My Bloch Spheres")
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    def test_plot_bloch_multivector_reverse_bits(self):
        """``reverse_bits=True`` flips the left-to-right subplot order."""
        img_ref = path_to_diagram_reference("bloch_multivector_reverse_bits.png")
        fig = plot_bloch_multivector(self.state, reverse_bits=True)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
