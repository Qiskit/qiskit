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

"""Chart axis generators.

A collection of functions that generate drawings from formatted input data.
See py:mod:`qiskit.visualization.pulse_v2.types` for more info on the required data.

In this module the input data is `types.ChartAxis`.

An end-user can write arbitrary functions that generate custom drawings.
Generators in this module are called with the `formatter` and `device` kwargs.
These data provides stylesheet configuration and backend system configuration.

The format of generator is restricted to:

    ```python

    def my_object_generator(data: ChartAxis,
                            formatter: Dict[str, Any],
                            device: DrawerBackendInfo) -> List[ElementaryData]:
        pass
    ```

Arbitrary generator function satisfying the above format can be accepted.
Returned `ElementaryData` can be arbitrary subclasses that are implemented in
the plotter API.
"""
from typing import Dict, Any, List

from qiskit.visualization.pulse_v2 import drawings, types, device_info


def gen_baseline(
    data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.LineData]:
    """Generate the baseline associated with the chart.

    Stylesheets:
        - The `baseline` style is applied.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `LineData` drawings.
    """
    style = {
        "alpha": formatter["alpha.baseline"],
        "zorder": formatter["layer.baseline"],
        "linewidth": formatter["line_width.baseline"],
        "linestyle": formatter["line_style.baseline"],
        "color": formatter["color.baseline"],
    }

    baseline = drawings.LineData(
        data_type=types.LineType.BASELINE,
        channels=data.channels,
        xvals=[types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT],
        yvals=[0, 0],
        ignore_scaling=True,
        styles=style,
    )

    return [baseline]


def gen_chart_name(
    data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the name of chart.

    Stylesheets:
        - The `axis_label` style is applied.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {
        "zorder": formatter["layer.axis_label"],
        "color": formatter["color.axis_label"],
        "size": formatter["text_size.axis_label"],
        "va": "center",
        "ha": "right",
    }

    text = drawings.TextData(
        data_type=types.LabelType.CH_NAME,
        channels=data.channels,
        xvals=[types.AbstractCoordinate.LEFT],
        yvals=[0],
        text=data.name,
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_chart_scale(
    data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the current scaling value of the chart.

    Stylesheets:
        - The `axis_label` style is applied.
        - The `annotate` style is partially applied for the font size.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {
        "zorder": formatter["layer.axis_label"],
        "color": formatter["color.axis_label"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "right",
    }

    scale_val = f"x{types.DynamicString.SCALE}"

    text = drawings.TextData(
        data_type=types.LabelType.CH_INFO,
        channels=data.channels,
        xvals=[types.AbstractCoordinate.LEFT],
        yvals=[-formatter["label_offset.chart_info"]],
        text=scale_val,
        ignore_scaling=True,
        styles=style,
    )

    return [text]


def gen_channel_freqs(
    data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo
) -> List[drawings.TextData]:
    """Generate the frequency values of associated channels.

    Stylesheets:
        - The `axis_label` style is applied.
        - The `annotate` style is partially applied for the font size.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {
        "zorder": formatter["layer.axis_label"],
        "color": formatter["color.axis_label"],
        "size": formatter["text_size.annotate"],
        "va": "center",
        "ha": "right",
    }

    if len(data.channels) > 1:
        sources = []
        for chan in data.channels:
            freq = device.get_channel_frequency(chan)
            if not freq:
                continue
            sources.append(f"{chan.name.upper()}: {freq / 1e9:.2f} GHz")
        freq_text = ", ".join(sources)
    else:
        freq = device.get_channel_frequency(data.channels[0])
        if freq:
            freq_text = f"{freq / 1e9:.2f} GHz"
        else:
            freq_text = ""

    text = drawings.TextData(
        data_type=types.LabelType.CH_INFO,
        channels=data.channels,
        xvals=[types.AbstractCoordinate.LEFT],
        yvals=[-formatter["label_offset.chart_info"]],
        text=freq_text or "no freq.",
        ignore_scaling=True,
        styles=style,
    )

    return [text]
