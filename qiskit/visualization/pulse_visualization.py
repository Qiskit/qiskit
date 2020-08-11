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
from typing import Union, Callable, List, Dict, Tuple

from qiskit.pulse import Schedule, Instruction, SamplePulse, Waveform, ScheduleComponent
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse.qcstyle import PulseStyle, SchedStyle
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse import matplotlib as _matplotlib

try:
    from matplotlib import get_backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def pulse_drawer(data: Union[Waveform, ScheduleComponent],
                 dt: int = 1,
                 style: Union[PulseStyle, SchedStyle] = None,
                 filename: str = None,
                 interp_method: Callable = None,
                 scale: float = None,
                 channel_scales: Dict[Channel, float] = None,
                 plot_all: bool = False,
                 plot_range: Tuple[Union[int, float], Union[int, float]] = None,
                 interactive: bool = False,
                 table: bool = False,
                 label: bool = False,
                 framechange: bool = True,
                 channels: List[Channel] = None,
                 show_framechange_channels: bool = True
                 ):
    """Plot the interpolated envelope of pulse and schedule.

    Args:
        data: Pulse or schedule object to plot.
        dt: Time interval of samples. Pulses are visualized in the unit of
            cycle time if not provided.
        style: A style sheet to configure plot appearance.
            See :mod:`~qiskit.visualization.pulse.qcstyle` for more information.
        filename: Name required to save pulse image. The drawer just returns
            `matplot.Figure` object if not provided.
        interp_method: Interpolation function. Interpolation is disabled in default.
            See :mod:`~qiskit.visualization.pulse.interpolation` for more information.
        scale: Scaling of waveform amplitude. Pulses are automatically
            scaled channel by channel if not provided.
        channel_scales: Dictionary of scale factor for specific channels.
            Scale of channels not specified here is overwritten by `scale`.
        plot_all: When set `True` plot empty channels.
        plot_range: A tuple of time range to plot.
        interactive: When set `True` show the circuit in a new window.
            This depends on the matplotlib backend being used supporting this.
        table: When set `True` draw event table for supported commands.
        label: When set `True` draw label for individual instructions.
        framechange: When set `True` draw framechange indicators.
        channels: A list of channel names to plot.
            All non-empty channels are shown if not provided.
        show_framechange_channels: When set `True` plot channels
            with only framechange instructions.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure object for the pulse envelope.

    Example:
        This example shows how to visualize your pulse schedule.
        Pulse names are added to the plot, unimportant channels are removed
        and the time window is truncated to draw out U3 pulse sequence of interest.

        .. jupyter-execute::

            import numpy as np
            import qiskit
            from qiskit import pulse
            from qiskit.test.mock.backends.almaden import FakeAlmaden

            inst_map = FakeAlmaden().defaults().instruction_schedule_map

            sched = pulse.Schedule()
            sched += inst_map.get('u3', 0, np.pi, 0, np.pi)
            sched += inst_map.get('measure', list(range(20))) << sched.duration

            channels = [pulse.DriveChannel(0), pulse.MeasureChannel(0)]
            scales = {pulse.DriveChannel(0): 10}

            qiskit.visualization.pulse_drawer(sched,
                                              channels=channels,
                                              plot_range=(0, 1000),
                                              label=True,
                                              channel_scales=scales)

        You are also able to call visualization module from the instance method::

            sched.draw(channels=channels, plot_range=(0, 1000), label=True, channel_scales=scales)

        To customize the format of the schedule plot, you can setup your style sheet.

        .. jupyter-execute::

            import numpy as np
            import qiskit
            from qiskit import pulse
            from qiskit.test.mock.backends.almaden import FakeAlmaden

            inst_map = FakeAlmaden().defaults().instruction_schedule_map

            sched = pulse.Schedule()
            sched += inst_map.get('u3', 0, np.pi, 0, np.pi)
            sched += inst_map.get('measure', list(range(20))) << sched.duration

            # setup style sheet
            my_style = qiskit.visualization.SchedStyle(
                figsize = (10, 5),
                bg_color='w',
                d_ch_color = ['#32cd32', '#556b2f'])

            channels = [pulse.DriveChannel(0), pulse.MeasureChannel(0)]
            scales = {pulse.DriveChannel(0): 10}

            qiskit.visualization.pulse_drawer(sched, style=my_style,
                                              channels=channels,
                                              plot_range=(0, 1000),
                                              label=True,
                                              channel_scales=scales)

    Raises:
        VisualizationError: when invalid data is given
        ImportError: when matplotlib is not installed
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if isinstance(data, (SamplePulse, Waveform)):
        drawer = _matplotlib.WaveformDrawer(style=style)
        image = drawer.draw(data, dt=dt, interp_method=interp_method, scale=scale)
    elif isinstance(data, (Schedule, Instruction)):
        drawer = _matplotlib.ScheduleDrawer(style=style)
        image = drawer.draw(data, dt=dt, interp_method=interp_method, scale=scale,
                            channel_scales=channel_scales, plot_range=plot_range,
                            plot_all=plot_all, table=table, label=label,
                            framechange=framechange, channels=channels,
                            show_framechange_channels=show_framechange_channels)
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
