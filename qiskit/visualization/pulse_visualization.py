# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
matplotlib pulse visualization.
"""

import logging

from qiskit.pulse import Schedule, Instruction, SamplePulse
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization import matplotlib as _matplotlib

logger = logging.getLogger(__name__)


def pulse_drawer(data, dt=1, style=None, filename=None, output='mpl',
                 interp_method=None, scaling=None, channels_to_plot=None,
                 plot_all=False, plot_range=None, interactive=False,
                 legend=True, table=True, label=False, framechange=True):
    """Plot the interpolated envelope of pulse

    Args:
        data (ScheduleComponent or SamplePulse): Data to plot
        dt (float): Time interval of samples
        style (OPStylePulse or OPStyleSched): A style sheet to configure plot appearance
        filename (str): Name required to save pulse image
        output (str): Select the output method to use for drawing the pulse object.
            Valid choices are `mpl`.
        interp_method (Callable): interpolation function
            See `qiskit.visualization.interpolation` for more information
        scaling (float): scaling of waveform amplitude
        channels_to_plot (list): A list of channel names to plot
        plot_all (bool): Plot empty channels
        plot_range (tuple): A tuple of time range to plot
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this)
        legend (bool): Draw Legend for supported commands
        table (bool): Draw event table for supported commands
        label (bool): Label individual instructions
        framechange (bool): Add framechange indicators
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope
    Raises:
        VisualizationError: when invalid data is given or lack of information
    """
    if output == 'mpl':
        if isinstance(data, SamplePulse):
            drawer = _matplotlib.SamplePulseDrawer(style=style)
            image = drawer.draw(data, dt=dt, interp_method=interp_method, scaling=scaling)
        elif isinstance(data, (Schedule, Instruction)):
            drawer = _matplotlib.ScheduleDrawer(style=style)
            image = drawer.draw(data, dt=dt, interp_method=interp_method, scaling=scaling,
                                plot_range=plot_range, channels_to_plot=channels_to_plot,
                                plot_all=plot_all, legend=legend, table=table, label=label)
        else:
            raise VisualizationError('This data cannot be visualized.')
    else:
        raise VisualizationError(
            'Invalid output type %s selected. The only valid choice '
            'is mpl' % output)

    if filename:
        image.savefig(filename, dpi=drawer.style.dpi, bbox_inches='tight')

    if image and interactive:
        image.show()
    return image
