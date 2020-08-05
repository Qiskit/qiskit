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

r"""
A collection of functions that generate drawing objects from formatted input data.
See py:mod:`qiskit.visualization.pulse_v2.data_types` for the detail of input data.


Framework
~~~~~~~~~
The functions in this module are generators for drawing objects. All drawing objects are created
by these generators. A stylesheet provides a list of generators and the core drawing
function calls the generators for each input instruction.

An end-user can write arbitrary functions with the following function signature:

    ```python
    def my_object_generator(inst_data: InstructionTuple) -> List[drawing_objects.TextData]:
        texts = []
        # create some text data
        text.append(TextData(data_type='custom', channel=inst_data.inst.channel, ...))

        return texts
    ```

The user-defined drawing object is created by adding the generator to the stylesheet:

    ```python
    my_custom_style = {'generator.waveform': [my_object_generator, ...]}
    ```

The user can set the custom stylesheet to the drawer interface.
"""

import re
from fractions import Fraction
from typing import Dict, Tuple, Any, List, Union

import numpy as np

from qiskit import pulse
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawing_objects, types, PULSE_STYLE


# Waveform related information generation


def _parse_waveform(inst_data: types.InstructionTuple) \
        -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r"""A helper function that generates sample data array of the waveform with
    instruction meta data.

    Args:
        inst_data: Instruction data set

    Raises:
        VisualizationError: When invalid instruction type is loaded.

    Returns:
        A tuple of xy data and metadata dictionary.
    """
    inst = inst_data.inst

    meta = dict()
    if isinstance(inst, instructions.Play):
        # pulse
        if isinstance(inst.pulse, pulse.ParametricPulse):
            pulse_data = inst.pulse.get_sample_pulse()
            meta.update(inst.pulse.parameters)
        else:
            pulse_data = inst.pulse
        xdata = np.arange(pulse_data.duration) + inst_data.t0
        ydata = pulse_data.samples
    elif isinstance(inst, instructions.Delay):
        # delay
        xdata = np.arange(inst.duration) + inst_data.t0
        ydata = np.zeros(inst.duration)
    elif isinstance(inst, instructions.Acquire):
        # acquire
        xdata = np.arange(inst.duration) + inst_data.t0
        ydata = np.ones(inst.duration)
        acq_data = {'memory slot': inst.mem_slot.name,
                    'register slot': inst.reg_slot.name if inst.reg_slot else 'N/A',
                    'discriminator': inst.discriminator.name if inst.discriminator else 'N/A',
                    'kernel': inst.kernel.name if inst.kernel else 'N/A'}
        meta.update(acq_data)
    else:
        raise VisualizationError('Instruction %s cannot be drawn by filled envelope.' % type(inst))

    meta.update({'duration (cycle time)': inst.duration,
                 'duration (sec)': inst.duration * inst_data.dt if inst_data.dt else 'N/A',
                 't0 (cycle time)': inst_data.t0,
                 't0 (sec)': inst_data.t0 * inst_data.dt if inst_data.dt else 'N/A',
                 'phase': inst_data.frame.phase,
                 'frequency': inst_data.frame.freq,
                 'name': inst.name})

    return xdata, ydata, meta


def _fill_waveform_color(channel: pulse.channels.Channel) \
        -> types.ComplexColors:
    r"""A helper function that returns color code of the fill waveform.

    Args:
        channel: Pulse channel object associated with the fill waveform.

    Raises:
        VisualizationError: When invalid channel is specified.

    Returns:
        A color code of real and imaginary part of the waveform.
    """
    if isinstance(channel, pulse.DriveChannel):
        colors = PULSE_STYLE['formatter.color.fill_waveform_d']
        if isinstance(colors, (tuple, list)):
            colors = types.ComplexColors(*colors)
        return colors
    if isinstance(channel, pulse.ControlChannel):
        colors = PULSE_STYLE['formatter.color.fill_waveform_u']
        if isinstance(colors, (tuple, list)):
            colors = types.ComplexColors(*colors)
        return colors
    if isinstance(channel, pulse.MeasureChannel):
        colors = PULSE_STYLE['formatter.color.fill_waveform_m']
        if isinstance(colors, (tuple, list)):
            colors = types.ComplexColors(*colors)
        return colors
    if isinstance(channel, pulse.AcquireChannel):
        colors = PULSE_STYLE['formatter.color.fill_waveform_a']
        if isinstance(colors, (tuple, list)):
            colors = types.ComplexColors(*colors)
        return colors

    raise VisualizationError('Channel type %s is not supported.' % type(channel))


def gen_filled_waveform_stepwise(inst_data: types.InstructionTuple) \
        -> List[drawing_objects.FilledAreaData]:
    r"""Generate filled area object of waveform envelope.

    The curve of envelope is not interpolated and presented as stepwise function.
    The `fill_waveform` style is applied.

    Args:
        inst_data: Waveform instruction data to draw.

    Returns:
        List of `FilledAreaData` drawing objects.
    """
    fill_objs = []

    # generate waveform data
    xdata, ydata, meta = _parse_waveform(inst_data)

    if PULSE_STYLE['formatter.control.apply_phase_modulation']:
        ydata = np.array(ydata, dtype=np.complex) * np.exp(1j * inst_data.frame.phase)
    xdata = np.concatenate((xdata, [xdata[-1] + 1]))
    ydata = np.repeat(ydata, 2)
    re_y = np.real(ydata)
    im_y = np.imag(ydata)
    time = np.concatenate(([xdata[0]], np.repeat(xdata[1:-1], 2), [xdata[-1]]))

    # setup style options
    channel = inst_data.inst.channel

    style = {'alpha': PULSE_STYLE['formatter.alpha.fill_waveform'],
             'zorder': PULSE_STYLE['formatter.layer.fill_waveform'],
             'linewidth': PULSE_STYLE['formatter.line_width.fill_waveform'],
             'linestyle': PULSE_STYLE['formatter.line_style.fill_waveform']}
    color = _fill_waveform_color(channel)

    # create real part
    if np.any(re_y):
        re_style = style.copy()
        re_style['color'] = color.real
        re_meta = meta.copy()
        re_meta['data'] = 'real'
        real = drawing_objects.FilledAreaData(data_type='Waveform',
                                              channel=channel,
                                              x=time,
                                              y1=re_y,
                                              y2=np.zeros_like(time),
                                              meta=re_meta,
                                              styles=re_style)
        fill_objs.append(real)

    # create imaginary part
    if np.any(im_y):
        im_style = style.copy()
        im_style['color'] = color.imaginary
        im_meta = meta.copy()
        im_meta['data'] = 'imaginary'
        imag = drawing_objects.FilledAreaData(data_type='Waveform',
                                              channel=channel,
                                              x=time,
                                              y1=im_y,
                                              y2=np.zeros_like(time),
                                              meta=im_meta,
                                              styles=im_style)
        fill_objs.append(imag)

    return fill_objs


def gen_iqx_latex_waveform_name(inst_data: types.InstructionTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate formatted instruction name associated with the waveform.

    Channel name and id are removed and the rotation angle is expressed in units of pi.
    CR pulse name is also converted with the controlled rotation angle divided by 2.

    Note that in many scientific articles the controlled rotation angle implies
    the actual rotation angle, but in IQX backend the rotation angle represents
    the difference between rotation angles with different control qubit states.

    For example:
        - 'X90p_d0_abcdefg' is converted into 'X(\frac{\pi}{2})'
        - 'CR90p_u0_abcdefg` is converted into 'CR(\frac{\pi}{4})'

    The `annotate` style is applied.

    Notes:
        This generator can convert pulse names used in the IQX backends.
        If pulses are provided by the third party providers or the user defined,
        the generator output may be the as-is pulse name.

    Args:
        inst_data: Waveform instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """
    systematic_name = inst_data.inst.pulse.name

    style = {'zorder': PULSE_STYLE['formatter.layer.annotate'],
             'color': PULSE_STYLE['formatter.color.annotate'],
             'size': PULSE_STYLE['formatter.text_size.annotate'],
             'va': 'center',
             'ha': 'center'}

    template = r'(?P<op>[A-Z]+)(?P<angle>[0-9]+)?(?P<sign>[pm])_(?P<ch>[dum])[0-9]+'
    match_result = re.match(template, systematic_name)
    if match_result is not None:
        match_dict = match_result.groupdict()
        sign = '' if match_dict['sign'] == 'p' else '-'
        if match_dict['op'] == 'CR':
            # cross resonance
            if match_dict['ch'] == 'u':
                op_name = r'{\rm CR}'
            else:
                op_name = r'\overline{\rm CR}'
            # IQX name def is not standard. Echo CR is annotated with pi/4 rather than pi/2
            angle_val = match_dict['angle']
            frac = Fraction(int(int(angle_val)/2), 180)
            if frac.numerator == 1:
                angle = r'\frac{{\pi}}{{{}}}'.format(frac.denominator)
            else:
                angle = r'\frac{{{}}}{{{}}}\pi'.format(frac.numerator, frac.denominator)
        else:
            # single qubit pulse
            op_name = r'{{\rm {}}}'.format(match_dict['op'])
            angle_val = match_dict['angle']
            if angle_val is None:
                angle = r'\pi'
            else:
                frac = Fraction(int(angle_val), 180)
                if frac.numerator == 1:
                    angle = r'\frac{{\pi}}{{{}}}'.format(frac.denominator)
                else:
                    angle = r'\frac{{{}}}{{{}}}\pi'.format(frac.numerator, frac.denominator)
        latex_name = r'{}({}{})'.format(op_name, sign, angle)
    else:
        if isinstance(inst_data.inst, pulse.instructions.Acquire):
            systematic_name = 'Acquire'
        if isinstance(inst_data.inst.channel, pulse.channels.MeasureChannel):
            systematic_name = 'Measure'
        latex_name = None

    text = drawing_objects.TextData(data_type='Waveform',
                                    channel=inst_data.inst.channel,
                                    x=inst_data.t0 + 0.5 * inst_data.inst.duration,
                                    y=PULSE_STYLE['formatter.label_offset.pulse_name'],
                                    text=systematic_name,
                                    latex=latex_name,
                                    styles=style)

    return [text]


# Channel related information generation


def gen_baseline(channel_data: types.ChannelTuple) \
        -> List[drawing_objects.LineData]:
    r"""Generate baseline associated with the channel.

    The `baseline` style is applied.

    Args:
        channel_data: Channel data to draw.

    Returns:
        List of `LineData` drawing objects.
    """
    style = {'alpha': PULSE_STYLE['formatter.alpha.baseline'],
             'zorder': PULSE_STYLE['formatter.layer.baseline'],
             'linewidth': PULSE_STYLE['formatter.line_width.baseline'],
             'linestyle': PULSE_STYLE['formatter.line_style.baseline'],
             'color': PULSE_STYLE['formatter.color.baseline']}

    baseline = drawing_objects.LineData(data_type='BaseLine',
                                        channel=channel_data.channel,
                                        x=None,
                                        y=0,
                                        styles=style)

    return [baseline]


def gen_latex_channel_name(channel_data: types.ChannelTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate channel name of provided channel.

    The `axis_label` style is applied.

    Args:
        channel_data: Channel data to draw.

    Returns:
        List of `TextData` drawing objects.
    """
    style = {'zorder': PULSE_STYLE['formatter.layer.axis_label'],
             'color': PULSE_STYLE['formatter.color.axis_label'],
             'size': PULSE_STYLE['formatter.text_size.axis_label'],
             'va': 'center',
             'ha': 'right'}
    latex_name = r'{}_{}'.format(channel_data.channel.prefix.upper(),
                                 channel_data.channel.index)

    text = drawing_objects.TextData(data_type='ChannelInfo',
                                    channel=channel_data.channel,
                                    x=0,
                                    y=0,
                                    text=channel_data.channel.name.upper(),
                                    latex=latex_name,
                                    styles=style)

    return [text]


def gen_scaling_info(channel_data: types.ChannelTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate channel scaling factor of provided channel.

    The `axis_label` style is applied.
    The `annotate` style is partially applied for the font size.

    Args:
        channel_data: Channel data to draw.

    Returns:
        List of `TextData` drawing objects.
    """
    if channel_data.scaling == 1:
        return []

    style = {'zorder': PULSE_STYLE['formatter.layer.axis_label'],
             'color': PULSE_STYLE['formatter.color.axis_label'],
             'size': PULSE_STYLE['formatter.text_size.annotate'],
             'va': 'center',
             'ha': 'right'}
    value = r'x{:.1f}'.format(channel_data.scaling)

    text = drawing_objects.TextData(data_type='ChannelInfo',
                                    channel=channel_data.channel,
                                    x=0,
                                    y=PULSE_STYLE['formatter.label_offset.scale_factor'],
                                    text=value,
                                    styles=style)

    return [text]


# Frame related information generation


def gen_latex_vz_label(frame_data: types.InstructionTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate formatted virtual Z rotations from provided frame instruction.

    Rotation angle is expressed in units of pi.
    If the denominator of fraction is larger than 10, the angle is expressed in units of radian.

    For example:
        - A value -3.14 is converted into `VZ(\pi)`
        - A value 1.57 is converted into `VZ(-\frac{\pi}{2})`
        - A value 0.123 is converted into `VZ(-0.123 rad.)`

    - The `frame_change` style is applied.
    - The `annotate` style is applied for font size.

    Notes:
        The phase operand of `PhaseShift` instruction has opposite sign to the Z gate definition.
        Thus the sign of rotation angle is inverted.

    Args:
        frame_data: Frame instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """
    _max_denom = 10

    style = {'zorder': PULSE_STYLE['formatter.layer.frame_change'],
             'color': PULSE_STYLE['formatter.color.frame_change'],
             'size': PULSE_STYLE['formatter.text_size.annotate'],
             'va': 'center',
             'ha': 'center'}

    frac = Fraction(np.abs(frame_data.frame.phase) / np.pi)
    if frac.denominator > _max_denom:
        angle = r'{:.2e}~{{\rm rad.}}'.format(frame_data.frame.phase)
    else:
        if frac.numerator == 1:
            if frac.denominator == 1:
                angle = r'\pi'
            else:
                angle = r'\frac{{\pi}}{{{}}}'.format(frac.denominator)
        else:
            angle = r'\frac{{{}}}{{{}}}\pi'.format(frac.numerator, frac.denominator)

    # Phase Shift is defined as negative value
    sign = '' if frame_data.frame.phase <= 0 else '-'

    text = drawing_objects.TextData(data_type='FrameInfo',
                                    channel=frame_data.inst[0].channel,
                                    x=frame_data.t0,
                                    y=PULSE_STYLE['formatter.label_offset.frame_change'],
                                    text=r'VZ({:.2f} rad.)'.format(-frame_data.frame.phase),
                                    latex=r'{{\rm VZ}}({}{})'.format(sign, angle),
                                    styles=style)

    return [text]


def gen_latex_frequency_mhz_value(frame_data: types.InstructionTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate formatted frequency change from provided frame instruction.

    Frequency change is expressed in units of MHz.

    For example:
        - A value 1,234,567 is converted into `\Delta f = 1.23 MHz`

    - The `frame_change` style is applied.
    - The `annotate` style is applied for font size.

    Args:
        frame_data: Frame instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """

    style = {'zorder': PULSE_STYLE['formatter.layer.frame_change'],
             'color': PULSE_STYLE['formatter.color.frame_change'],
             'size': PULSE_STYLE['formatter.text_size.annotate'],
             'va': 'center',
             'ha': 'center'}

    text_df = u'\u0394' + 'f={:.2f} MHz'.format(frame_data.frame.freq/1e6)
    latex_df = r'\Delta f = {:.2f} ~{{\rm MHz}}'.format(frame_data.frame.freq/1e6)

    text = drawing_objects.TextData(data_type='FrameInfo',
                                    channel=frame_data.inst[0].channel,
                                    x=frame_data.t0,
                                    y=PULSE_STYLE['formatter.label_offset.frame_change'],
                                    text=text_df,
                                    latex=latex_df,
                                    styles=style)

    return [text]


def gen_raw_frame_operand_values(frame_data: types.InstructionTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate both phase and frequency change from provided frame instruction.

    Frequency change is expressed in scientific notation.

    For example:
        - A phase change 1.57 and frequency change 1,234,567 are written by `(1.57, 1.2e+06)`

    - The `frame_change` style is applied.
    - The `annotate` style is applied for font size.

    Args:
        frame_data: Frame instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """

    style = {'zorder': PULSE_STYLE['formatter.layer.frame_change'],
             'color': PULSE_STYLE['formatter.color.frame_change'],
             'size': PULSE_STYLE['formatter.text_size.annotate'],
             'va': 'center',
             'ha': 'center'}

    frame_info = '({:.2f}, {:.1e})'.format(frame_data.frame.phase, frame_data.frame.freq)

    text = drawing_objects.TextData(data_type='FrameInfo',
                                    channel=frame_data.inst[0].channel,
                                    x=frame_data.t0,
                                    y=PULSE_STYLE['formatter.label_offset.frame_change'],
                                    text=frame_info,
                                    styles=style)

    return [text]


def gen_frame_symbol(frame_data: types.InstructionTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate a frame change symbol with instruction meta data from provided frame instruction.

    The `frame_change` style is applied.

    The symbol type in unicode is specified in `formatter.unicode_symbol.frame_change`.
    The symbol type in latex is specified in `formatter.latex_symbol.frame_change`.

    Args:
        frame_data: Frame instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """

    style = {'zorder': PULSE_STYLE['formatter.layer.frame_change'],
             'color': PULSE_STYLE['formatter.color.frame_change'],
             'size': PULSE_STYLE['formatter.text_size.frame_change'],
             'va': 'center',
             'ha': 'center'}

    program = []
    for inst in frame_data.inst:
        if isinstance(inst, (instructions.SetFrequency, instructions.ShiftFrequency)):
            program.append('{}({:.2e} Hz)'.format(inst.__class__.__name__, inst.frequency))
        elif isinstance(inst, (instructions.SetPhase, instructions.ShiftPhase)):
            program.append('{}({:.2f} rad.)'.format(inst.__class__.__name__, inst.phase))

    meta = {'total phase change': frame_data.frame.phase,
            'total frequency change': frame_data.frame.freq,
            'program': program,
            't0 (cycle time)': frame_data.t0,
            't0 (sec)': frame_data.t0 * frame_data.dt if frame_data.dt else 'N/A'}

    uni_symbol = PULSE_STYLE['formatter.unicode_symbol.frame_change']
    latex = PULSE_STYLE['formatter.latex_symbol.frame_change']

    text = drawing_objects.TextData(data_type='Symbol',
                                    channel=frame_data.inst[0].channel,
                                    x=frame_data.t0,
                                    y=0,
                                    text=uni_symbol,
                                    latex=latex,
                                    meta=meta,
                                    styles=style)

    return [text]


# Misc information generation


def gen_snapshot_symbol(misc_data: types.NonPulseTuple) \
        -> List[drawing_objects.TextData]:
    r"""Generate a snapshot symbol with instruction meta data from provided snapshot instruction.

    The snapshot symbol is capped by the snapshot label.

    - The `snapshot` style is applied for snapshot symbol.
    - The `annotate` style is applied for label font size.

    The symbol type in unicode is specified in `formatter.unicode_symbol.snapshot`.
    The symbol type in latex is specified in `formatter.latex_symbol.snapshot`.

    Args:
        misc_data: Snapshot instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """

    if not isinstance(misc_data.inst, pulse.instructions.Snapshot):
        return []

    symbol_style = {'zorder': PULSE_STYLE['formatter.layer.snapshot'],
                    'color': PULSE_STYLE['formatter.color.snapshot'],
                    'size': PULSE_STYLE['formatter.text_size.snapshot'],
                    'va': 'bottom',
                    'ha': 'center'}

    label_style = {'zorder': PULSE_STYLE['formatter.layer.snapshot'],
                   'color': PULSE_STYLE['formatter.color.snapshot'],
                   'size': PULSE_STYLE['formatter.text_size.annotate'],
                   'va': 'bottom',
                   'ha': 'center'}

    meta = {'snapshot type': misc_data.inst.type,
            't0 (cycle time)': misc_data.t0,
            't0 (sec)': misc_data.t0 * misc_data.dt if misc_data.dt else 'N/A'}

    uni_symbol = PULSE_STYLE['formatter.unicode_symbol.snapshot']
    latex = PULSE_STYLE['formatter.latex_symbol.snapshot']

    symbol_text = drawing_objects.TextData(data_type='Symbol',
                                           channel=misc_data.inst.channel,
                                           x=misc_data.t0,
                                           y=0,
                                           text=uni_symbol,
                                           latex=latex,
                                           meta=meta,
                                           styles=symbol_style)

    label_text = drawing_objects.TextData(data_type='Symbol',
                                          channel=misc_data.inst.channel,
                                          x=misc_data.t0,
                                          y=PULSE_STYLE['formatter.label_offset.snapshot'],
                                          text=misc_data.inst.label,
                                          styles=label_style)

    return [symbol_text, label_text]


def gen_barrier(misc_data: types.NonPulseTuple) \
        -> List[Union[drawing_objects.LineData, drawing_objects.TextData]]:
    r"""Generate a barrier from provided relative barrier instruction..

    The `barrier` style is applied.

    Args:
        misc_data: Snapshot instruction data to draw.

    Returns:
        List of `TextData` drawing objects.
    """

    if not isinstance(misc_data.inst, pulse.instructions.RelativeBarrier):
        return []

    style = {'alpha': PULSE_STYLE['formatter.alpha.barrier'],
             'zorder': PULSE_STYLE['formatter.layer.barrier'],
             'linewidth': PULSE_STYLE['formatter.line_width.barrier'],
             'linestyle': PULSE_STYLE['formatter.line_style.barrier'],
             'color': PULSE_STYLE['formatter.color.barrier']}

    lines = []
    for chan in misc_data.inst.channels:
        line = drawing_objects.LineData(data_type='Barrier',
                                        channel=chan,
                                        x=misc_data.t0,
                                        y=None,
                                        styles=style)
        lines.append(line)

    return lines
