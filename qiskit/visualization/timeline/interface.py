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

from qiskit import circuit
from qiskit.visualization.timeline.events import InstructionDurations
from qiskit.visualization.timeline import drawer_style, types, core, styles
from qiskit.visualization.exceptions import VisualizationError

from typing import Optional, Dict, Any, List


def timeline_drawer(scheduled_circuit: circuit.QuantumCircuit,
                    inst_durations: InstructionDurations,
                    stylesheet: Optional[Dict[str, Any]] = styles.IqxStandard,
                    backend: Optional[str] = 'mpl',
                    filename: Optional[str] = None,
                    bits: Optional[List[types.Bits]] = None,
                    show_idle: Optional[bool] = None,
                    show_clbits: Optional[bool] = None,
                    show_barriers:  Optional[bool] = None,
                    show_delays:  Optional[bool] = None,
                    ax: Optional = None):
    """User interface of timeline drawer.

    Args:
        scheduled_circuit: Input circuit to draw.
            This program should be transpiled with gate time information
            ahead of visualization.
        inst_durations: A table of duration for specific gate instructions.
        stylesheet: A dictionary of timeline drawer stylesheet.
            See below for details.
        backend: A string to specify the plotter to draw timeline.
            `matplotlib` is used as a default plotter.
        filename: If provided the output image is dumped into a file under the filename.
        bits: List of bits to draw.
            Timelines of unspecified bits are removed if provided.
        show_idle: A control property to show idle timeline.
            Set `True` to show timeline without gate instructions.
        show_clbits: A control property to show classical bits.
            Set `True` to show classical bits.
        show_barriers: A control property to show barrier instructions.
            Set `True` to show barrier instructions.
        show_delays: A control property to show delay instructions.
            Set `True` to show delay instructions.
        ax (matplotlib.axes.Axes): An optional Axes object to be used as a canvas.
            If not provided a new matplotlib figure will be created and returned.
            This is only valid when the `backend` is set to `mpl`.

    Returns:
        Image data. The data format depends on the specified backend.

    Examples:
        todo: add jupyter execute

    Raises:
        VisualizationError: When invalid backend is specified.

    Stylesheet:
        todo: add stylesheet options
    """
    # update stylesheet
    drawer_style.update(stylesheet)
    drawer_style.current_stylesheet = stylesheet.__class__.__name__

    # update control properties
    if show_idle is not None:
        drawer_style['formatter.control.show_idle'] = show_idle
    if show_clbits is not None:
        drawer_style['formatter.control.show_clbits'] = show_clbits
    if show_barriers is not None:
        drawer_style['formatter.control.show_barriers'] = show_barriers
    if show_delays is not None:
        drawer_style['formatter.control.show_delays'] = show_delays

    # setup data container
    ddc = core.DrawDataContainer()
    ddc.load_program(scheduled_circuit=scheduled_circuit,
                     inst_durations=inst_durations)
    ddc.update_preference(visible_bits=bits)

    # draw
    if backend == 'mpl':
        # matplotlib plotter
        try:
            from qiskit.visualization.timeline.backends.matplotlib import MplPlotter
            from matplotlib import pyplot as plt, get_backend
        except ImportError:
            raise VisualizationError('Matplotlib is not installed. ' 
                                     'Try pip install matplotlib to use this format.')

        plotter = MplPlotter(draw_data=ddc, axis=ax)
        plotter.draw_data()

        if ax is None:
            if filename:
                plotter.figure.savefig(filename,
                                       dpi=drawer_style['formatter.general.dpi'],
                                       bbox_inches='tight')
            if get_backend() in ['module://ipykernel.pylab.backend_inline', 'nbAgg']:
                plt.close(plotter.figure)

            return plotter.figure
    else:
        VisualizationError('Backend {backend} is not supported.'.format(backend=backend))
