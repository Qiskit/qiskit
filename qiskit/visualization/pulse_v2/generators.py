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
Generator function of drawing IRs.
"""

import numpy as np
from collections import OrderedDict

from qiskit import pulse
from typing import Callable, Union, Dict, Tuple, Any, List
from qiskit.visualization.pulse_v2.events import PhaseFreqTuple
from qiskit.visualization.pulse_v2 import drawing_objects
from qiskit.visualization.pulse_v2 import style_lib

from qiskit import pulse

from qiskit.visualization.exceptions import VisualizationError


def _parse_waveform(
        t0: int,
        frame: PhaseFreqTuple,
        inst: Union[pulse.instructions.Play,
                    pulse.instructions.Delay,
                    pulse.instructions.Acquire]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate waveform data and instruction metadata dictionary.

    Args:
        t0: The time when the instruction is issued.
        frame: Phase and frequency at the associated time.
        inst: Instruction data.

    Raises:
        VisualizationError: When invalid instruction type is loaded.
    """

    meta = OrderedDict()
    if isinstance(inst, pulse.instructions.Play):
        # pulse
        if isinstance(inst.pulse, pulse.ParametricPulse):
            pulse_data = inst.pulse.get_sample_pulse()
            meta.update(inst.pulse.parameters)
        else:
            pulse_data = inst.pulse
        xdata = np.arange(pulse_data.duration) + t0
        ydata = pulse_data.samples
    elif isinstance(inst, pulse.instructions.Delay):
        # delay
        xdata = np.arange(inst.duration) + t0
        ydata = np.zeros(inst.duration)
    elif isinstance(inst, pulse.instructions.Acquire):
        # acquire
        xdata = np.arange(inst.duration) + t0
        ydata = np.ones(inst.duration)
        acq_data = {
            'memory slot': inst.mem_slot.name,
            'register slot': inst.reg_slot.name if inst.reg_slot else 'not assigned',
            'discriminator': inst.discriminator.name if inst.discriminator else 'not assigned',
            'kernel': inst.kernel.name if inst.kernel else 'not assigned'
        }
        meta.update(acq_data)
    else:
        raise VisualizationError('Instruction %s cannot be drawn by filled envelope.' % type(inst))

    meta.update({
        'duration': inst.duration,
        'phase': frame.phase,
        'frequency': frame.freq,
        'name': inst.name
    })

    return xdata, ydata, meta


def gen_filled_waveform_stepwise(
        t0: int,
        frame: PhaseFreqTuple,
        inst: Union[pulse.instructions.Play,
                    pulse.instructions.Delay,
                    pulse.instructions.Acquire],
        **formatter
) -> List[drawing_objects.FilledAreaData]:
    """Generate filled area object of pulse envelope."""
    xdata, ydata, meta = _parse_waveform(t0, frame, inst)

    if formatter['option']['phase_modulation']:
        ydata *= np.exp(1j * frame.phase)

    ydata = np.repeat(ydata, 2)
    re_y = np.real(ydata)
    im_y = np.imag(ydata)
    time = np.concatenate((xdata[0], np.repeat(xdata[1, -1], 2), xdata[-1]))

    if isinstance(inst.channel, pulse.DriveChannel):
        color = formatter['color']['ch_d']
    elif isinstance(inst.channel, pulse.ControlChannel):
        color = formatter['color']['ch_u']
    elif isinstance(inst.channel, pulse.MeasureChannel):
        color = formatter['color']['ch_m']
    elif isinstance(inst.channel, pulse.AcquireChannel):
        color = formatter['color']['ch_a']
    else:
        raise VisualizationError('Channel type %s is not supported.' % type(inst.channel))

    style = {
        'alpha': formatter['alpha']['waveform'],
        'zorder': formatter['layer']['waveform'],
        'linewidth': formatter['line_width']['waveform']
    }

    objs = []

    # create real part
    if any(re_y):
        re_style = style.copy()
        re_style['color'] = color.real
        re_meta = meta.copy()
        re_meta['data'] = 'real'
        real = drawing_objects.FilledAreaData(
            data_type='WaveForm',
            channel=inst.channel,
            x=time,
            y1=re_y,
            y2=np.zeros_like(time),
            meta=re_meta,
            offset=0,
            visible=True,
            styles=re_style
        )
        objs.append(real)

    # create imaginary part
    if any(im_y):
        im_style = style.copy()
        im_style['color'] = color.imag
        im_meta = meta.copy()
        im_meta['data'] = 'imaginary'
        imag = drawing_objects.FilledAreaData(
            data_type='WaveForm',
            channel=inst.channel,
            x=time,
            y1=im_y,
            y2=np.zeros_like(time),
            meta=im_meta,
            offset=0,
            visible=True,
            styles=im_style
        )
        objs.append(imag)

    return objs


def gen_baseline(
        t0: int,
        t1: int,
        channel: pulse.channels.Channel,
        **formatter
) -> List[drawing_objects.LineData]:
    """Generate baseline."""
    style = {
        'alpha': formatter['alpha']['baseline'],
        'zorder': formatter['layer']['baseline'],
        'linewidth': formatter['line_width']['baseline'],
        'color': formatter['color']['baseline']
    }

    baseline = drawing_objects.LineData(
        data_type='BaseLine',
        channel=channel,
        x=np.array([t0, t1]),
        y=np.array([0, 0]),
        meta=None,
        offset=0,
        visible=True,
        styles=style
    )

    return [baseline]


def gen_latex_channel_name(
        channel: pulse.channels.Channel,
        **formatter
):
    style = {
        'zorder': formatter['layer']['axis_label'],
        'color': formatter['color']['axis_label'],
        'size': formatter['text_size']['axis_label'],
        'va': 'center',
        'ha': 'right'
    }
    name = r'${}_{}$'.format(channel.prefix.upper(), channel.index)

    text = drawing_objects.TextData(
        data_type='ChannelName',
        channel=channel,
        x=-formatter['margin']['left'],
        y=0,
        text=name,
        meta=None,
        offset=0,
        visible=True,
        styles=style
    )

    return [text]












