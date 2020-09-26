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
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import types, core, stylesheet


def draw(program: circuit.QuantumCircuit,
         style: Optional[Dict[str, Any]] = None,
         time_range: Tuple[int, int] = None,
         disable_bits: List[types.Bits] = None,
         show_clbits: Optional[bool] = None,
         show_idle: Optional[bool] = None,
         show_barriers: Optional[bool] = None,
         show_delays: Optional[bool] = None,
         show_labels: bool = True,
         plotter: Optional[str] = types.Plotter.MPL.value,
         axis: Optional[Any] = None,
         filename: Optional[str] = None):
    """Generate visualization data for pulse programs.

    Args:
        program: Program to visualize.
            This program should be transpiled with gate time information before visualization.
        style: Stylesheet options. This can be dictionary or preset stylesheet classes. See
            :py:class:~`qiskit.visualization.timeline.stylesheets.IqxStandard`,
            :py:class:~`qiskit.visualization.timeline.stylesheets.IqxSimple`, and
            :py:class:~`qiskit.visualization.timeline.stylesheets.IqxDebugging` for details of
            preset stylesheets. See also the stylesheet section for details of configuration keys.
        time_range: Set horizontal axis limit.
        disable_bits: List of qubits of classical bits not shown in the output image.
        show_clbits: A control property to show classical bits.
            Set `True` to show classical bits.
        show_idle: A control property to show idle timeline.
            Set `True` to show timeline without gates.
        show_barriers: A control property to show barrier instructions.
            Set `True` to show barrier instructions.
        show_delays: A control property to show delay instructions.
            Set `True` to show delay instructions.
        show_labels: A control property to show annotations, i.e. name, of gates.
            Set `True` to show annotations.
        plotter: Name of plotter API to generate an output image.
            See plotter section for details.
        axis: Arbitrary object passed to the plotter. If this object is provided,
            the plotters uses given `axis` instead of internally initializing a figure object.
            This object format depends on the plotter. See plotters section for details.
        filename: If provided the output image is dumped into a file under the filename.

    Returns:
        Image data. The generated data format depends on the `plotter`.
        If matplotlib family is specified, this will be a `matplotlib.pyplot.Figure` data.

    Examples:
        todo: add jupyter execute

    Plotters:
        - `mpl`: Matplotlib API to generate 2D image. Charts are placed along y axis with
            vertical offset. This API takes matplotlib.axes.Axes as `axis` input.

    Stylesheet:
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

    Raises:
        VisualizationError: When invalid backend is specified.

    """
    # update stylesheet
    temp_style = stylesheet.QiskitTimelineStyle()
    temp_style.update(style or stylesheet.IqxStandard())

    # update control properties
    if show_idle is not None:
        temp_style['formatter.control.show_idle'] = show_idle

    if show_clbits is not None:
        temp_style['formatter.control.show_clbits'] = show_clbits

    if show_barriers is not None:
        temp_style['formatter.control.show_barriers'] = show_barriers

    if show_delays is not None:
        temp_style['formatter.control.show_delays'] = show_delays

    # create empty canvas and load program
    canvas = core.DrawerCanvas(stylesheet=temp_style)
    canvas.load_program(program=program)

    #
    # update configuration
    #

    # time range
    if time_range:
        canvas.set_time_range(*time_range)

    # bits not shown
    if disable_bits:
        for bit in disable_bits:
            canvas.set_disable_bits(bit, remove=True)

    # show labels
    if not show_labels:
        labels = [types.DrawingLabel.DELAY,
                  types.DrawingLabel.GATE_PARAM,
                  types.DrawingLabel.GATE_NAME]
        for label in labels:
            canvas.set_disable_type(label, remove=True)

    canvas.update()

    #
    # Call plotter API and generate image
    #

    if plotter == types.Plotter.MPL.value:
        try:
            from qiskit.visualization.timeline.plotters import MplPlotter
        except ImportError:
            raise ImportError('Must have Matplotlib installed.')

        plotter_api = MplPlotter(canvas=canvas, axis=axis)
        plotter_api.draw()
    else:
        raise VisualizationError('Plotter API {name} is not supported.'.format(name=plotter))

    # save figure
    if filename:
        plotter_api.save_file(filename=filename)

    return plotter_api.get_image()
