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

# pylint: disable=unused-argument

"""Waveform generators.

A collection of functions that generate drawings from formatted input data.
See py:mod:`qiskit.visualization.pulse_v2.types` for more info on the required data.

In this module the input data is `types.PulseInstruction`.

An end-user can write arbitrary functions that generate custom drawings.
Generators in this module are called with the `formatter` and `device` kwargs.
These data provides stylesheet configuration and backend system configuration.

The format of generator is restricted to:

    ```python

    def my_object_generator(data: PulseInstruction,
                            formatter: Dict[str, Any],
                            device: DrawerBackendInfo) -> List[ElementaryData]:
        pass
    ```

Arbitrary generator function satisfying the above format can be accepted.
Returned `ElementaryData` can be arbitrary subclasses that are implemented in
the plotter API.
"""

from __future__ import annotations
import re
from fractions import Fraction
from typing import Any

import numpy as np

from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info


def gen_filled_waveform_stepwise(
    data: types.PulseInstruction, formatter: dict[str, Any], device: device_info.DrawerBackendInfo
) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """Generate filled area objects of the real and the imaginary part of waveform envelope.

    The curve of envelope is not interpolated nor smoothed and presented
    as stepwise function at each data point.

    Stylesheets:
        - The `fill_waveform` style is applied.

    Args:
        data: Waveform instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `LineData`, `BoxData`, or `TextData` drawings.

    Raises:
        VisualizationError: When the instruction parser returns invalid data format.
    """
    # generate waveform data
    waveform_data = _parse_waveform(data)
    channel = data.inst.channel

    # update metadata
    meta = waveform_data.meta
    qind = device.get_qubit_index(channel)
    meta.update({"qubit": qind if qind is not None else "N/A"})

    if isinstance(waveform_data, types.ParsedInstruction):
        # Draw waveform with fixed shape

        xdata = waveform_data.xvals
        ydata = waveform_data.yvals

        # phase modulation
        if formatter["control.apply_phase_modulation"]:
            ydata = np.asarray(ydata, dtype=complex) * np.exp(1j * data.frame.phase)
        else:
            ydata = np.asarray(ydata, dtype=complex)

        return _draw_shaped_waveform(
            xdata=xdata, ydata=ydata, meta=meta, channel=channel, formatter=formatter
        )

    elif isinstance(waveform_data, types.OpaqueShape):
        # Draw parametric pulse with unbound parameters

        # parameter name
        unbound_params = []
        for pname, pval in data.inst.pulse.parameters.items():
            if isinstance(pval, circuit.ParameterExpression):
                unbound_params.append(pname)

        pulse_data = data.inst.pulse
        if isinstance(pulse_data, library.SymbolicPulse):
            pulse_shape = pulse_data.pulse_type
        else:
            pulse_shape = "Waveform"

        return _draw_opaque_waveform(
            init_time=data.t0,
            duration=waveform_data.duration,
            pulse_shape=pulse_shape,
            pnames=unbound_params,
            meta=meta,
            channel=channel,
            formatter=formatter,
        )

    else:
        raise VisualizationError("Invalid data format is provided.")


def gen_ibmq_latex_waveform_name(
    data: types.PulseInstruction, formatter: dict[str, Any], device: device_info.DrawerBackendInfo
) -> list[drawings.TextData]:
    r"""Generate the formatted instruction name associated with the waveform.

    Channel name and ID string are removed and the rotation angle is expressed in units of pi.
    The controlled rotation angle associated with the CR pulse name is divided by 2.

    Note that in many scientific articles the controlled rotation angle implies
    the actual rotation angle, but in IQX backend the rotation angle represents
    the difference between rotation angles with different control qubit states.

    For example:
        - 'X90p_d0_abcdefg' is converted into 'X(\frac{\pi}{2})'
        - 'CR90p_u0_abcdefg` is converted into 'CR(\frac{\pi}{4})'

    Stylesheets:
        - The `annotate` style is applied.

    Notes:
        This generator can convert pulse names used in the IQX backends.
        If pulses are provided by the third party providers or the user defined,
        the generator output may be the as-is pulse name.

    Args:
        data: Waveform instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    if data.is_opaque:
        return []

    style = {
        "zorder": formatter["layer.annotate"],
        "color": formatter["color.annotate"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "center",
    }

    if isinstance(data.inst, pulse.instructions.Acquire):
        systematic_name = "Acquire"
        latex_name = None
    elif isinstance(data.inst, instructions.Delay):
        systematic_name = data.inst.name or "Delay"
        latex_name = None
    else:
        pulse_data = data.inst.pulse
        if pulse_data.name:
            systematic_name = pulse_data.name
        else:
            if isinstance(pulse_data, library.SymbolicPulse):
                systematic_name = pulse_data.pulse_type
            else:
                systematic_name = "Waveform"

        template = r"(?P<op>[A-Z]+)(?P<angle>[0-9]+)?(?P<sign>[pm])_(?P<ch>[dum])[0-9]+"
        match_result = re.match(template, systematic_name)
        if match_result is not None:
            match_dict = match_result.groupdict()
            sign = "" if match_dict["sign"] == "p" else "-"
            if match_dict["op"] == "CR":
                # cross resonance
                if match_dict["ch"] == "u":
                    op_name = r"{\rm CR}"
                else:
                    op_name = r"\overline{\rm CR}"
                # IQX name def is not standard. Echo CR is annotated with pi/4 rather than pi/2
                angle_val = match_dict["angle"]
                frac = Fraction(int(int(angle_val) / 2), 180)
                if frac.numerator == 1:
                    angle = rf"\pi/{frac.denominator:d}"
                else:
                    angle = r"{num:d}/{denom:d} \pi".format(
                        num=frac.numerator, denom=frac.denominator
                    )
            else:
                # single qubit pulse
                op_name = r"{{\rm {}}}".format(match_dict["op"])
                angle_val = match_dict["angle"]
                if angle_val is None:
                    angle = r"\pi"
                else:
                    frac = Fraction(int(angle_val), 180)
                    if frac.numerator == 1:
                        angle = rf"\pi/{frac.denominator:d}"
                    else:
                        angle = r"{num:d}/{denom:d} \pi".format(
                            num=frac.numerator, denom=frac.denominator
                        )
            latex_name = rf"{op_name}({sign}{angle})"
        else:
            latex_name = None

    text = drawings.TextData(
        data_type=types.LabelType.PULSE_NAME,
        channels=data.inst.channel,
        xvals=[data.t0 + 0.5 * data.inst.duration],
        yvals=[-formatter["label_offset.pulse_name"]],
        text=systematic_name,
        latex=latex_name,
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_waveform_max_value(
    data: types.PulseInstruction, formatter: dict[str, Any], device: device_info.DrawerBackendInfo
) -> list[drawings.TextData]:
    """Generate the annotation for the maximum waveform height for
    the real and the imaginary part of the waveform envelope.

    Maximum values smaller than the vertical resolution limit is ignored.

    Stylesheets:
        - The `annotate` style is applied.

    Args:
        data: Waveform instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    if data.is_opaque:
        return []

    style = {
        "zorder": formatter["layer.annotate"],
        "color": formatter["color.annotate"],
        "size": formatter["text_size.annotate"],
        "ha": "center",
    }

    # only pulses.
    if isinstance(data.inst, instructions.Play):
        # pulse
        operand = data.inst.pulse
        if isinstance(operand, (pulse.ParametricPulse, pulse.SymbolicPulse)):
            pulse_data = operand.get_waveform()
        else:
            pulse_data = operand
        xdata = np.arange(pulse_data.duration) + data.t0
        ydata = pulse_data.samples
    else:
        return []

    # phase modulation
    if formatter["control.apply_phase_modulation"]:
        ydata = np.asarray(ydata, dtype=complex) * np.exp(1j * data.frame.phase)
    else:
        ydata = np.asarray(ydata, dtype=complex)

    texts = []

    # max of real part
    re_maxind = np.argmax(np.abs(ydata.real))
    if np.abs(ydata.real[re_maxind]) > 0.01:
        # generator shows only 2 digits after the decimal point.
        if ydata.real[re_maxind] > 0:
            max_val = f"{ydata.real[re_maxind]:.2f}\n\u25BE"
            re_style = {"va": "bottom"}
        else:
            max_val = f"\u25B4\n{ydata.real[re_maxind]:.2f}"
            re_style = {"va": "top"}
        re_style.update(style)
        re_text = drawings.TextData(
            data_type=types.LabelType.PULSE_INFO,
            channels=data.inst.channel,
            xvals=[xdata[re_maxind]],
            yvals=[ydata.real[re_maxind]],
            text=max_val,
            styles=re_style,
        )
        texts.append(re_text)

    # max of imag part
    im_maxind = np.argmax(np.abs(ydata.imag))
    if np.abs(ydata.imag[im_maxind]) > 0.01:
        # generator shows only 2 digits after the decimal point.
        if ydata.imag[im_maxind] > 0:
            max_val = f"{ydata.imag[im_maxind]:.2f}\n\u25BE"
            im_style = {"va": "bottom"}
        else:
            max_val = f"\u25B4\n{ydata.imag[im_maxind]:.2f}"
            im_style = {"va": "top"}
        im_style.update(style)
        im_text = drawings.TextData(
            data_type=types.LabelType.PULSE_INFO,
            channels=data.inst.channel,
            xvals=[xdata[im_maxind]],
            yvals=[ydata.imag[im_maxind]],
            text=max_val,
            styles=im_style,
        )
        texts.append(im_text)

    return texts


def _draw_shaped_waveform(
    xdata: np.ndarray,
    ydata: np.ndarray,
    meta: dict[str, Any],
    channel: pulse.channels.PulseChannel,
    formatter: dict[str, Any],
) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """A private function that generates drawings of stepwise pulse lines.

    Args:
        xdata: Array of horizontal coordinate of waveform envelope.
        ydata: Array of vertical coordinate of waveform envelope.
        meta: Metadata dictionary of the waveform.
        channel: Channel associated with the waveform to draw.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of drawings.

    Raises:
        VisualizationError: When the waveform color for channel is not defined.
    """
    fill_objs: list[drawings.LineData | drawings.BoxData | drawings.TextData] = []

    resolution = formatter["general.vertical_resolution"]

    # stepwise interpolation
    xdata: np.ndarray = np.concatenate((xdata, [xdata[-1] + 1]))
    ydata = np.repeat(ydata, 2)
    re_y = np.real(ydata)
    im_y = np.imag(ydata)
    time: np.ndarray = np.concatenate(([xdata[0]], np.repeat(xdata[1:-1], 2), [xdata[-1]]))

    # setup style options
    style = {
        "alpha": formatter["alpha.fill_waveform"],
        "zorder": formatter["layer.fill_waveform"],
        "linewidth": formatter["line_width.fill_waveform"],
        "linestyle": formatter["line_style.fill_waveform"],
    }

    try:
        color_real, color_imag = formatter["color.waveforms"][channel.prefix.upper()]
    except KeyError as ex:
        raise VisualizationError(
            f"Waveform color for channel type {channel.prefix} is not defined"
        ) from ex

    # create real part
    if np.any(re_y):
        # data compression
        re_valid_inds = _find_consecutive_index(re_y, resolution)
        # stylesheet
        re_style = {"color": color_real}
        re_style.update(style)
        # metadata
        re_meta = {"data": "real"}
        re_meta.update(meta)
        # active xy data
        re_xvals = time[re_valid_inds]
        re_yvals = re_y[re_valid_inds]

        # object
        real = drawings.LineData(
            data_type=types.WaveformType.REAL,
            channels=channel,
            xvals=re_xvals,
            yvals=re_yvals,
            fill=formatter["control.fill_waveform"],
            meta=re_meta,
            styles=re_style,
        )
        fill_objs.append(real)

    # create imaginary part
    if np.any(im_y):
        # data compression
        im_valid_inds = _find_consecutive_index(im_y, resolution)
        # stylesheet
        im_style = {"color": color_imag}
        im_style.update(style)
        # metadata
        im_meta = {"data": "imag"}
        im_meta.update(meta)
        # active xy data
        im_xvals = time[im_valid_inds]
        im_yvals = im_y[im_valid_inds]

        # object
        imag = drawings.LineData(
            data_type=types.WaveformType.IMAG,
            channels=channel,
            xvals=im_xvals,
            yvals=im_yvals,
            fill=formatter["control.fill_waveform"],
            meta=im_meta,
            styles=im_style,
        )
        fill_objs.append(imag)

    return fill_objs


def _draw_opaque_waveform(
    init_time: int,
    duration: int,
    pulse_shape: str,
    pnames: list[str],
    meta: dict[str, Any],
    channel: pulse.channels.PulseChannel,
    formatter: dict[str, Any],
) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """A private function that generates drawings of stepwise pulse lines.

    Args:
        init_time: Time when the opaque waveform starts.
        duration: Duration of opaque waveform. This can be None or ParameterExpression.
        pulse_shape: String that represents pulse shape.
        pnames: List of parameter names.
        meta: Metadata dictionary of the waveform.
        channel: Channel associated with the waveform to draw.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of drawings.
    """
    fill_objs: list[drawings.LineData | drawings.BoxData | drawings.TextData] = []

    fc, ec = formatter["color.opaque_shape"]
    # setup style options
    box_style = {
        "zorder": formatter["layer.fill_waveform"],
        "alpha": formatter["alpha.opaque_shape"],
        "linewidth": formatter["line_width.opaque_shape"],
        "linestyle": formatter["line_style.opaque_shape"],
        "facecolor": fc,
        "edgecolor": ec,
    }

    if duration is None or isinstance(duration, circuit.ParameterExpression):
        duration = formatter["box_width.opaque_shape"]

    box_obj = drawings.BoxData(
        data_type=types.WaveformType.OPAQUE,
        channels=channel,
        xvals=[init_time, init_time + duration],
        yvals=[
            -0.5 * formatter["box_height.opaque_shape"],
            0.5 * formatter["box_height.opaque_shape"],
        ],
        meta=meta,
        ignore_scaling=True,
        styles=box_style,
    )
    fill_objs.append(box_obj)

    # parameter name
    func_repr = "{func}({params})".format(func=pulse_shape, params=", ".join(pnames))

    text_style = {
        "zorder": formatter["layer.annotate"],
        "color": formatter["color.annotate"],
        "size": formatter["text_size.annotate"],
        "va": "bottom",
        "ha": "center",
    }

    text_obj = drawings.TextData(
        data_type=types.LabelType.OPAQUE_BOXTEXT,
        channels=channel,
        xvals=[init_time + 0.5 * duration],
        yvals=[0.5 * formatter["box_height.opaque_shape"]],
        text=func_repr,
        ignore_scaling=True,
        styles=text_style,
    )

    fill_objs.append(text_obj)

    return fill_objs


def _find_consecutive_index(data_array: np.ndarray, resolution: float) -> np.ndarray:
    """A helper function to return non-consecutive index from the given list.

    This drastically reduces memory footprint to represent a drawing,
    especially for samples of very long flat-topped Gaussian pulses.
    Tiny value fluctuation smaller than `resolution` threshold is removed.

    Args:
        data_array: The array of numbers.
        resolution: Minimum resolution of sample values.

    Returns:
        The compressed data array.
    """
    try:
        vector = np.asarray(data_array, dtype=float)
        diff = np.diff(vector)
        diff[np.where(np.abs(diff) < resolution)] = 0
        # keep left and right edges
        consecutive_l = np.insert(diff.astype(bool), 0, True)
        consecutive_r = np.append(diff.astype(bool), True)
        return consecutive_l | consecutive_r

    except ValueError:
        return np.ones_like(data_array).astype(bool)


def _parse_waveform(
    data: types.PulseInstruction,
) -> types.ParsedInstruction | types.OpaqueShape:
    """A helper function that generates an array for the waveform with
    instruction metadata.

    Args:
        data: Instruction data set

    Raises:
        VisualizationError: When invalid instruction type is loaded.

    Returns:
        A data source to generate a drawing.
    """
    inst = data.inst

    meta: dict[str, Any] = {}
    if isinstance(inst, instructions.Play):
        # pulse
        operand = inst.pulse
        if isinstance(operand, (pulse.ParametricPulse, pulse.SymbolicPulse)):
            # parametric pulse
            params = operand.parameters
            duration = params.pop("duration", None)
            if isinstance(duration, circuit.Parameter):
                duration = None

            if isinstance(operand, library.SymbolicPulse):
                pulse_shape = operand.pulse_type
            else:
                pulse_shape = "Waveform"
            meta["waveform shape"] = pulse_shape

            meta.update(
                {
                    key: val.name if isinstance(val, circuit.Parameter) else val
                    for key, val in params.items()
                }
            )
            if data.is_opaque:
                # parametric pulse with unbound parameter
                if duration:
                    meta.update(
                        {
                            "duration (cycle time)": inst.duration,
                            "duration (sec)": inst.duration * data.dt if data.dt else "N/A",
                        }
                    )
                else:
                    meta.update({"duration (cycle time)": "N/A", "duration (sec)": "N/A"})

                meta.update(
                    {
                        "t0 (cycle time)": data.t0,
                        "t0 (sec)": data.t0 * data.dt if data.dt else "N/A",
                        "phase": data.frame.phase,
                        "frequency": data.frame.freq,
                        "name": inst.name,
                    }
                )

                return types.OpaqueShape(duration=duration, meta=meta)
            else:
                # fixed shape parametric pulse
                pulse_data = operand.get_waveform()
        else:
            # waveform
            pulse_data = operand
        xdata = np.arange(pulse_data.duration) + data.t0
        ydata = pulse_data.samples
    elif isinstance(inst, instructions.Delay):
        # delay
        xdata = np.arange(inst.duration) + data.t0
        ydata = np.zeros(inst.duration)
    elif isinstance(inst, instructions.Acquire):
        # acquire
        xdata = np.arange(inst.duration) + data.t0
        ydata = np.ones(inst.duration)
        acq_data = {
            "memory slot": inst.mem_slot.name,
            "register slot": inst.reg_slot.name if inst.reg_slot else "N/A",
            "discriminator": inst.discriminator.name if inst.discriminator else "N/A",
            "kernel": inst.kernel.name if inst.kernel else "N/A",
        }
        meta.update(acq_data)
    else:
        raise VisualizationError(
            "Unsupported instruction {inst} by "
            "filled envelope.".format(inst=inst.__class__.__name__)
        )

    meta.update(
        {
            "duration (cycle time)": inst.duration,
            "duration (sec)": inst.duration * data.dt if data.dt else "N/A",
            "t0 (cycle time)": data.t0,
            "t0 (sec)": data.t0 * data.dt if data.dt else "N/A",
            "phase": data.frame.phase,
            "frequency": data.frame.freq,
            "name": inst.name,
        }
    )

    return types.ParsedInstruction(xvals=xdata, yvals=ydata, meta=meta)
