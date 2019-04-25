# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Tests for comparing the outputs of circuit drawer with expected ones."""

import os
import unittest

from qiskit.pulse import Schedule, pulse_lib
from qiskit.tools.visualization import HAS_MATPLOTLIB, pulse_drawer

from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestPulseVisualizationImplementation(QiskitVisualizationTestCase):
    """Visual accuracy of visualization tools outputs tests."""

    schedule_matplotlib_reference = path_to_diagram_reference('pulse_schedule_matplotlib_ref.png')
    pulse_matplotlib_reference = path_to_diagram_reference('pulse_matplotlib_ref.png')

    def sample_schedule(self):
        """Generate a sample schedule that includes the most common elements of
           pulse schedules.
        """
        schedule = Schedule()

        return schedule

    def sample_pulse(self):
        return pulse_lib.gaussian(20, 0.8, 1.0, name='test')

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    def test_schedule_matplotlib_drawer(self):
        filename = self._get_resource_path('current_matplot.png')
        sched = self.sample_schedule()
        pulse_drawer(sched, filename=filename, output='mpl')
        self.assertImagesAreEqual(filename, self.schedule_matplotlib_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    def test_pulse_matplotlib_drawer(self):
        filename = self._get_resource_path('current_matplot.png')
        pulse = self.sample_pulse()
        pulse_drawer(pulse, filename=filename, output='mpl')
        self.assertImagesAreEqual(filename, self.pulse_matplotlib_reference)
        os.remove(filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
