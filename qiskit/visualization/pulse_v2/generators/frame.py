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

"""Frame change generators.

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
from fractions import Fraction
from typing import Dict, Any, List, Tuple

import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info


def gen_formatted_phase(
    data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label from provided frame instruction.

    Rotation angle is expressed in units of pi.
    If the denominator of fraction is larger than 10, the angle is expressed in units of radian.

    For example:
        - A value -3.14 is converted into `VZ(\\pi)`
        - A value 1.57 is converted into `VZ(-\\frac{\\pi}{2})`
        - A value 0.123 is converted into `VZ(-0.123 rad.)`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Notes:
        The phase operand of `PhaseShift` instruction has opposite sign to the Z gate definition.
        Thus the sign of rotation angle is inverted.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    _max_denom = 10

    style = {
        "zorder": formatter["layer.frame_change"],
        "color": formatter["color.frame_change"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "center",
    }

    plain_phase, latex_phase = _phase_to_text(
        formatter=formatter, phase=data.frame.phase, max_denom=_max_denom, flip=True
    )

    text = drawings.TextData(
        data_type=types.LabelType.FRAME,
        channels=data.inst[0].channel,
        xvals=[data.t0],
        yvals=[formatter["label_offset.frame_change"]],
        text=f"VZ({plain_phase})",
        latex=fr"{{\rm VZ}}({latex_phase})",
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_formatted_freq_mhz(
    data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the formatted frequency change label from provided frame instruction.

    Frequency change is expressed in units of MHz.

    For example:
        - A value 1,234,567 is converted into `\\Delta f = 1.23 MHz`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    _unit = "MHz"

    style = {
        "zorder": formatter["layer.frame_change"],
        "color": formatter["color.frame_change"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "center",
    }

    plain_freq, latex_freq = _freq_to_text(formatter=formatter, freq=data.frame.freq, unit=_unit)

    text = drawings.TextData(
        data_type=types.LabelType.FRAME,
        channels=data.inst[0].channel,
        xvals=[data.t0],
        yvals=[formatter["label_offset.frame_change"]],
        text=f"\u0394f = {plain_freq}",
        latex=fr"\Delta f = {latex_freq}",
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_formatted_frame_values(
    data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label and the frequency change label
    from provided frame instruction.

    Phase value is placed on top of the symbol, and frequency value is placed below the symbol.
    See :py:func:`gen_formatted_phase` and :py:func:`gen_formatted_freq_mhz` for details.

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    texts = []

    _max_denom = 10
    _unit = "MHz"

    style = {
        "zorder": formatter["layer.frame_change"],
        "color": formatter["color.frame_change"],
        "size": formatter["text_size.annotate"],
        "ha": "center",
    }

    # phase value
    if data.frame.phase != 0:
        plain_phase, latex_phase = _phase_to_text(
            formatter=formatter, phase=data.frame.phase, max_denom=_max_denom, flip=True
        )
        phase_style = {"va": "center"}
        phase_style.update(style)

        phase = drawings.TextData(
            data_type=types.LabelType.FRAME,
            channels=data.inst[0].channel,
            xvals=[data.t0],
            yvals=[formatter["label_offset.frame_change"]],
            text=f"VZ({plain_phase})",
            latex=fr"{{\rm VZ}}({latex_phase})",
            ignore_scaling=True,
            styles=phase_style,
        )
        texts.append(phase)

    # frequency value
    if data.frame.freq != 0:
        plain_freq, latex_freq = _freq_to_text(
            formatter=formatter, freq=data.frame.freq, unit=_unit
        )
        freq_style = {"va": "center"}
        freq_style.update(style)

        freq = drawings.TextData(
            data_type=types.LabelType.FRAME,
            channels=data.inst[0].channel,
            xvals=[data.t0],
            yvals=[2 * formatter["label_offset.frame_change"]],
            text=f"\u0394f = {plain_freq}",
            latex=fr"\Delta f = {latex_freq}",
            ignore_scaling=True,
            styles=freq_style,
        )
        texts.append(freq)

    return texts


def gen_raw_operand_values_compact(
    data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label and the frequency change label
    from provided frame instruction.

    Raw operand values are shown in compact form. Frequency change is expressed
    in scientific notation. Values are shown in two lines.

    For example:
        - A phase change 1.57 and frequency change 1,234,567 are written by `1.57\\n1.2e+06`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """

    style = {
        "zorder": formatter["layer.frame_change"],
        "color": formatter["color.frame_change"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "center",
    }

    if data.frame.freq == 0:
        freq_sci_notation = "0.0"
    else:
        abs_freq = np.abs(data.frame.freq)
        freq_sci_notation = "{base:.1f}e{exp:d}".format(
            base=data.frame.freq / (10 ** int(np.floor(np.log10(abs_freq)))),
            exp=int(np.floor(np.log10(abs_freq))),
        )
    frame_info = f"{data.frame.phase:.2f}\n{freq_sci_notation}"

    text = drawings.TextData(
        data_type=types.LabelType.FRAME,
        channels=data.inst[0].channel,
        xvals=[data.t0],
        yvals=[1.2 * formatter["label_offset.frame_change"]],
        text=frame_info,
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_frame_symbol(
    data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate a frame change symbol with instruction meta data from provided frame instruction.

    Stylesheets:
        - The `frame_change` style is applied.
        - The symbol type in unicode is specified in `formatter.unicode_symbol.frame_change`.
        - The symbol type in latex is specified in `formatter.latex_symbol.frame_change`.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    if data.frame.phase == 0 and data.frame.freq == 0:
        return []

    style = {
        "zorder": formatter["layer.frame_change"],
        "color": formatter["color.frame_change"],
        "size": formatter["text_size.frame_change"],
        "va": "center",
        "ha": "center",
    }

    program = []
    for inst in data.inst:
        if isinstance(inst, (instructions.SetFrequency, instructions.ShiftFrequency)):
            try:
                program.append(f"{inst.__class__.__name__}({inst.frequency:.2e} Hz)")
            except TypeError:
                # parameter expression
                program.append(f"{inst.__class__.__name__}({inst.frequency})")
        elif isinstance(inst, (instructions.SetPhase, instructions.ShiftPhase)):
            try:
                program.append(f"{inst.__class__.__name__}({inst.phase:.2f} rad.)")
            except TypeError:
                # parameter expression
                program.append(f"{inst.__class__.__name__}({inst.phase})")

    meta = {
        "total phase change": data.frame.phase,
        "total frequency change": data.frame.freq,
        "program": ", ".join(program),
        "t0 (cycle time)": data.t0,
        "t0 (sec)": data.t0 * data.dt if data.dt else "N/A",
    }

    text = drawings.TextData(
        data_type=types.SymbolType.FRAME,
        channels=data.inst[0].channel,
        xvals=[data.t0],
        yvals=[0],
        text=formatter["unicode_symbol.frame_change"],
        latex=formatter["latex_symbol.frame_change"],
        ignore_scaling=True,
        meta=meta,
        styles=style,
    )

    return [text]


def _phase_to_text(
    formatter: Dict[str, Any], phase: float, max_denom: int = 10, flip: bool = True
) -> Tuple[str, str]:
    """A helper function to convert a float value to text with pi.

    Args:
        formatter: Dictionary of stylesheet settings.
        phase: A phase value in units of rad.
        max_denom: Maximum denominator. Return raw value if exceed.
        flip: Set `True` to flip the sign.

    Returns:
        Standard text and latex text of phase value.
    """
    try:
        phase = float(phase)
    except TypeError:
        # unbound parameter
        return (
            formatter["unicode_symbol.phase_parameter"],
            formatter["latex_symbol.phase_parameter"],
        )

    frac = Fraction(np.abs(phase) / np.pi)

    if phase == 0:
        return "0", r"0"

    num = frac.numerator
    denom = frac.denominator
    if denom > max_denom:
        # denominator is too large
        latex = fr"{np.abs(phase):.2f}"
        plain = f"{np.abs(phase):.2f}"
    else:
        if num == 1:
            if denom == 1:
                latex = r"\pi"
                plain = "pi"
            else:
                latex = fr"\pi/{denom:d}"
                plain = f"pi/{denom:d}"
        else:
            latex = fr"{num:d}/{denom:d} \pi"
            plain = f"{num:d}/{denom:d} pi"

    if flip:
        sign = "-" if phase > 0 else ""
    else:
        sign = "-" if phase < 0 else ""

    return sign + plain, sign + latex


def _freq_to_text(formatter: Dict[str, Any], freq: float, unit: str = "MHz") -> Tuple[str, str]:
    """A helper function to convert a freq value to text with supplementary unit.

    Args:
        formatter: Dictionary of stylesheet settings.
        freq: A frequency value in units of Hz.
        unit: Supplementary unit. THz, GHz, MHz, kHz, Hz are supported.

    Returns:
        Standard text and latex text of phase value.

    Raises:
        VisualizationError: When unsupported unit is specified.
    """
    try:
        freq = float(freq)
    except TypeError:
        # unbound parameter
        return formatter["unicode_symbol.freq_parameter"], formatter["latex_symbol.freq_parameter"]

    unit_table = {"THz": 1e12, "GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1}

    try:
        value = freq / unit_table[unit]
    except KeyError as ex:
        raise VisualizationError(f"Unit {unit} is not supported.") from ex

    latex = fr"{value:.2f}~{{\rm {unit}}}"
    plain = f"{value:.2f} {unit}"

    return plain, latex
