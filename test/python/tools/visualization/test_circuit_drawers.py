# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for circuit_drawer."""

import unittest
from unittest.mock import patch

from PIL.Image import Image

import qiskit.tools.visualization._circuit_visualization as _cv
import qiskit.tools.visualization._matplotlib as _mpl

from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization import VisualizationError

from ...common import QiskitTestCase


class TestCircuitDrawer(QiskitTestCase):
    """Tests circuit_drawer upon different values of `output` parameter.

    In particular, it tests handling of the following cases:
    1) Correct value of `output` provided;
    2) Wrong value of `output` provided;
    3) No value of `output` provided.

    Also, this test case checks how `interactive=True` is handled.
    """

    def setUp(self):
        self.draw_methods = {
            'text': '_text_circuit_drawer',
            'latex': '_latex_circuit_drawer',
            'latex_source': '_generate_latex_source',
            'mpl': '_matplotlib_circuit_drawer'
        }

        self.interactive_draw_methods = ('latex', 'mpl')

    def test_correct_output_provided(self):
        """Tests calling of the correctly specified output circuit drawer.

        Since testing routine is the same for all four circuit drawers,
        test is split into subtests, one for corresponding circuit drawer.

        """
        # Create a subTest for current circuit drawer
        for draw_method in self.draw_methods:
            with self.subTest('Test calling of the {} draw method'.format(draw_method),
                              draw_method=draw_method):

                # Patch function corresponding to the current circuit drawer such that
                # it does nothing
                with patch.object(_cv, self.draw_methods[draw_method], return_value=None)\
                        as mock_draw_method:

                    # Check that corresponding function was called once
                    circuit_drawer(None, output=draw_method)
                    mock_draw_method.assert_called_once()

    def test_wrong_output_provided(self):
        """Tests correct exceptioning of wrong output format """
        with self.assertRaises(VisualizationError):
            circuit_drawer(None, output='wrong_output')

    @unittest.skip('Test for deprecated fallback is skipped ')
    def test_no_output_provided(self):
        """Tests how `circuit_drawer` falls back when no `output` is given

        First, it creates a subTest to check how `circuit_drawer` falls back
        to LaTeX circuit drawer.

        Next, it creates a subTest to test fallback to matplotlib circuit drawer.
        Here, it checks two cases: `HAS_MATPLOTLIB=True` and `HAS_MATPLOTLIB=False`.

        """
        # Create a subTest for fallback to latex_circuit_drawer
        with self.subTest('Test fallback to latex_circuit_drawer'):

            # Patch _latex_circuit_drawer such that it does nothing
            with patch.object(_cv, self.draw_methods['latex'], return_value=None)\
                    as mock_latex_circuit_drawer:

                # Check that _latex_circuit_drawer was called once
                circuit_drawer(None)
                mock_latex_circuit_drawer.assert_called_once()

        # Create a subTest for fallback to matplotlib_circuit_drawer
        with self.subTest('Test fallback to matplotlib_circuit_drawer'):

            # Patch _latex_circuit_drawer such that it raises FileNotFoundError
            with patch.object(_cv, self.draw_methods['latex'], side_effect=FileNotFoundError) as _:

                # Patch HAS_MATPLOTLIB attribute of _matplotlib to True
                with patch.object(_mpl, 'HAS_MATPLOTLIB', return_value=True):

                    # Patch _matplotlib_circuit_drawer such that it does nothing
                    with patch.object(_cv, self.draw_methods['mpl'])\
                            as mock_matplotlib_circuit_drawer:

                        # Check that _matplotlib_circuit_drawer was called once
                        circuit_drawer(None)
                        mock_matplotlib_circuit_drawer.assert_called_once()

                # Patch HAS_MATPLOTLIB attribute of _matplotlib to False
                with patch.object(_mpl, 'HAS_MATPLOTLIB', new=False):

                    # Check that circuit_drawer raises correct exception
                    with self.assertRaises(ImportError):
                        circuit_drawer(None)

    def test_interactive(self):
        """Tests that `interactive=True` makes Image to be shown

        Since testing routine is the same for both circuit drawers
        that support interactivity, test is split into subtests,
        one for corresponding circuit drawer.
        """
        # Create a subTest for current circuit drawer
        for draw_method in self.interactive_draw_methods:
            with self.subTest('Test interactive regime for {} output'.format(draw_method),
                              draw_method=draw_method):

                # Patch corresponding circuit_drawer such that it returns an instance of Image
                with patch.object(_cv, self.draw_methods[draw_method], return_value=Image()) as _:

                    # Patch show attribute of Image such that it does nothing
                    with patch.object(Image, 'show', return_value=None) as mock_show:

                        # Check that show was called once
                        circuit_drawer(None, output=draw_method, interactive=True)
                        mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
