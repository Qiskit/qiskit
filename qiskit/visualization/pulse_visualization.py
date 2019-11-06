# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
matplotlib pulse visualization.
"""

from qiskit.pulse import Schedule, Instruction, SamplePulse
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse import matplotlib as _matplotlib

if _matplotlib.HAS_MATPLOTLIB:
    from matplotlib import get_backend


def pulse_drawer(data, dt=1, style=None, filename=None,
                 interp_method=None, scaling=None, channels_to_plot=None,
                 plot_all=False, plot_range=None, interactive=False,
                 table=True, label=False, framechange=True):
    """Plot the interpolated envelope of pulse

    Args:
        data (ScheduleComponent or SamplePulse): Data to plot
        dt (float): Time interval of samples
        style (PulseStyle or SchedStyle): A style sheet to configure
            plot appearance
        filename (str): Name required to save pulse image
        interp_method (Callable): interpolation function
            See `qiskit.visualization.interpolation` for more information
        scaling (float): scaling of waveform amplitude
        channels_to_plot (list): A list of channel names to plot
        plot_all (bool): Plot empty channels
        plot_range (tuple): A tuple of time range to plot
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this)
        table (bool): Draw event table for supported commands
        label (bool): Label individual instructions
        framechange (bool): Add framechange indicators
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope
    Raises:
        VisualizationError: when invalid data is given or lack of information
        ImportError: when matplotlib is not installed
    """
    if not _matplotlib.HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if isinstance(data, SamplePulse):
        drawer = _matplotlib.SamplePulseDrawer(style=style)
        image = drawer.draw(data, dt=dt, interp_method=interp_method, scaling=scaling)
    elif isinstance(data, (Schedule, Instruction)):
        drawer = _matplotlib.ScheduleDrawer(style=style)
        image = drawer.draw(data, dt=dt, interp_method=interp_method, scaling=scaling,
                            plot_range=plot_range, channels_to_plot=channels_to_plot,
                            plot_all=plot_all, table=table, label=label,
                            framechange=framechange)
    else:
        raise VisualizationError('This data cannot be visualized.')

    if filename:
        image.savefig(filename, dpi=drawer.style.dpi, bbox_inches='tight')

    if get_backend() in ['module://ipykernel.pylab.backend_inline',
                         'nbAgg']:
        _matplotlib.plt.close(image)
    if image and interactive:
        image.show()
    return image
