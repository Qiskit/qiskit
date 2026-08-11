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

"""Tests for the Bloch sphere visualization."""

import unittest
from unittest import mock

from qiskit.utils import optionals

from .visualization import QiskitVisualizationTestCase


@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
class TestBloch(QiskitVisualizationTestCase):
    """Tests for the ``Bloch`` class."""

    def test_show_does_not_raise(self):
        """Regression test of gh-16741: ``show`` passed the figure positionally to
        ``pyplot.show``, which only accepts a keyword-only ``block`` argument and so
        raised ``TypeError`` on every backend."""
        from matplotlib import pyplot as plt

        from qiskit.visualization.bloch import Bloch

        bloch = Bloch()
        bloch.add_vectors([0, 0, 1])
        with mock.patch.object(plt, "show") as mocked_show:
            bloch.show()
        mocked_show.assert_called_once_with()
        self.assertIsNotNone(bloch.fig)
        plt.close(bloch.fig)
