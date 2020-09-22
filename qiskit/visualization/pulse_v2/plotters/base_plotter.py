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

"""Base plotter API."""

from abc import ABC, abstractmethod
from qiskit.visualization.pulse_v2 import core


class BasePlotter(ABC):

    def __init__(self, canvas: core.DrawerCanvas):
        """Create new plotter.

        Args:
            canvas: Configured drawer canvas object.
        """
        self.canvas = canvas

    @abstractmethod
    def initialize_canvas(self):
        """Format appearance of matplotlib canvas."""
        raise NotImplementedError

    @abstractmethod
    def draw(self):
        """Output drawing objects stored in canvas object."""
        raise NotImplementedError

    @abstractmethod
    def save_file(self, filename: str):
        """Save image to file.

        Args:
            filename: File path to output image data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_image(self):
        """Get image data to return."""
        raise NotImplementedError
