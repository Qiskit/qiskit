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

from typing import Optional, Dict, Any, List, Tuple

from qiskit import circuit
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawer_style, types, core, styles


def timeline_drawer(scheduled_circuit: circuit.QuantumCircuit,
                    inst_durations: InstructionDurations,
                    stylesheet: Optional[Dict[str, Any]] = None,
                    backend: Optional[str] = 'mpl',
                    filename: Optional[str] = None,
                    bits: Optional[List[types.Bits]] = None,
                    plot_range: Tuple[int, int] = None,
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
        plot_range: Tuple of numbers (t0, t1) that specify a time range to show.
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

    Stylesheet options:
        formatter.general.fig_unit_height: Height of output image in inch per unit.
        formatter.general.fig_width: Width of output image in inch.
        formatter.general.dpi: DPI of image when it's saved.
        formatter.margin.top: Top margin of output image.
        formatter.margin.bottom: Bottom margin of output image.
        formatter.margin.left_percent: Left margin of timeline.
            Value is percentage of entire timeline length.
        formatter.margin.right_percent: Right margin of timeline.
            Value is percentage of entire timeline length.
        formatter.margin.interval: Margin in between timelines.
        formatter.margin.link_interval_dt: Minimum horizontal spacing of bit links.
        formatter.time_bucket.edge_dt: Edge length of time buckets of gate instructions.
        formatter.color.background: Background color.
        formatter.color.timeslot: Face color of timeline.
        formatter.color.gate_name: Text color of gate name annotation.
        formatter.color.bit_name: Text color of bit label.
        formatter.color.barrier: Line color of barriers.
        formatter.box_height.gate: Height of time buckets of gate instructions.
        formatter.box_height.timeslot: Height of timelines.
        formatter.layer.gate: Layer index of time buckets of gate instructions.
        formatter.layer.timeslot: Layer index of timeslines.
        formatter.layer.gate_name: Layer index of gate name annotations.
        formatter.layer.bit_name: Layer index of bit label.
        formatter.layer.frame_change: Layer index of frame change symbol.
        formatter.layer.barrier. Layer index of barrier lines.
        formatter.layer.bit_link: Layer index of bit link lines.
        formatter.alpha.gate: Transparency of time buckets of gate instructions.
        formatter.alpha.timeslot: Transparency of timelines.
        formatter.alpha.barrier: Transparency of barrier lines.
        formatter.alpha.bit_link: Transparency of bit link lines.
        formatter.line_width.gate: Edge line width of time buckets of gate instructions.
        formatter.line_width.timeslot: Edge line width of timelines.
        formatter.line_width.barrier: Line width of barrier lines.
        formatter.line_width.bit_link: Line width of bit_link lines.
        formatter.line_style.barrier: Line style of barrier lines.
            The style syntax conforms to the matplotlib.
        formatter.line_style.bit_link: Line style of bit_link lines.
            The style syntax conforms to the matplotlib.
        formatter.font_size.gate_name: Font size of gate name annotations.
        formatter.font_size.bit_name: Font size of bit labels.
        formatter.font_size.frame_change: Font size of frame_change symbols.
        formatter.label_offset.frame_change: Vertical offset of frame change operand annotation
            from the center of the symbol.
        formatter.unicode_symbol.frame_change: Unicode representation of frame change symbol.
        formatter.latex_symbol.frame_change: Latex representation of frame change symbol.
        formatter.control.show_idle: Default control property to show idle bits.
            Set `True` to show idle bits.
        formatter.control.show_clbits: Default control property to show classical bits.
            Set `True` to show classical bits.
        formatter.control.show_barriers: Default control property to show barriers.
            Set `True` to show barriers.
        formatter.control.show_delays: Default control property to show delays.
            Set `True` to show delays.
        layout.gate_color: Layout callback function that takes gate name and
            returns color code.
            See :py:mod:`~qiskit.visualization.timeline.layouts` for details.
        layout.latex_gate_name: Layout callback function that takes gate name and
            convert it into latex format.
            See :py:mod:`~qiskit.visualization.timeline.layouts` for details.
        layout.bit_arrange: Layout callback function that takes list of bits and
            sort by index or bit types.
            See :py:mod:`~qiskit.visualization.timeline.layouts` for details.
        generator.gates: List of generator callback function that takes
            `ScheduledGate` object and returns drawing objects.
            See :py:mod:`~qiskit.visualization.timeline.generators` for details.
        generator.bits: List of generator callback function that takes
            a bit object and returns drawing objects.
            See :py:mod:`~qiskit.visualization.timeline.generators` for details.
        generator.barriers: List of generator callback function that takes
            `Barrier` object and returns drawing objects.
            See :py:mod:`~qiskit.visualization.timeline.generators` for details.
        generator.bit_links: List of generator callback function that takes
            `GateLink` object and returns drawing objects.
            See :py:mod:`~qiskit.visualization.timeline.generators` for details.
    """
    # update stylesheet
    drawer_style.update(stylesheet or styles.IqxStandard())
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

    # set time range
    if plot_range:
        ddc.set_time_range(t_start=plot_range[0], t_end=plot_range[1])

    # update objects
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
        plotter.draw()

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
