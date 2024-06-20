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

"""
A collection of functions that decide the layout of an output image.
See :py:mod:`~qiskit.visualization.pulse_v2.types` for more info on the required data.

There are 3 types of layout functions in this module.

1. layout.chart_channel_map

An end-user can write arbitrary functions that output the custom channel ordering
associated with group name. Layout function in this module are called with the
`formatter` and `device` kwargs. These data provides stylesheet configuration
and backend system configuration.

The layout function is restricted to:


    ```python
    def my_channel_layout(channels: List[pulse.channels.Channel],
                          formatter: Dict[str, Any],
                          device: DrawerBackendInfo
                          ) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
        ordered_channels = []
        # arrange order of channels

        for key, channels in my_ordering_dict.items():
            yield key, channels
    ```

2. layout.time_axis_map

An end-user can write arbitrary functions that output the `HorizontalAxis` data set that
will be later consumed by the plotter API to update the horizontal axis appearance.
Layout function in this module are called with the `time_window`, `axis_breaks`, and `dt` kwargs.
These data provides horizontal axis limit, axis break position, and time resolution, respectively.

See py:mod:`qiskit.visualization.pulse_v2.types` for more info on the required
data.

    ```python
    def my_horizontal_axis(time_window: Tuple[int, int],
                           axis_breaks: List[Tuple[int, int]],
                           dt: Optional[float] = None) -> HorizontalAxis:
        # write horizontal axis configuration

        return horizontal_axis
    ```

3. layout.figure_title

An end-user can write arbitrary functions that output the string data that
will be later consumed by the plotter API to output the figure title.
Layout functions in this module are called with the `program` and `device` kwargs.
This data provides input program and backend system configurations.

    ```python
    def my_figure_title(program: Union[pulse.Waveform, pulse.Schedule],
                        device: DrawerBackendInfo) -> str:

        return 'title'
    ```

An arbitrary layout function satisfying the above format can be accepted.
"""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union

import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo


def channel_type_grouped_sort(
    channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo
) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign single channel per chart. Channels are grouped by type and
    sorted by index in ascending order.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, D2, C0, C2, M0, M2, A0, A2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    chan_type_dict = defaultdict(list)

    for chan in channels:
        chan_type_dict[type(chan)].append(chan)

    ordered_channels = []

    # drive channels
    d_chans = chan_type_dict.get(pulse.DriveChannel, [])
    ordered_channels.extend(sorted(d_chans, key=lambda x: x.index))

    # control channels
    c_chans = chan_type_dict.get(pulse.ControlChannel, [])
    ordered_channels.extend(sorted(c_chans, key=lambda x: x.index))

    # measure channels
    m_chans = chan_type_dict.get(pulse.MeasureChannel, [])
    ordered_channels.extend(sorted(m_chans, key=lambda x: x.index))

    # acquire channels
    if formatter["control.show_acquire_channel"]:
        a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
        ordered_channels.extend(sorted(a_chans, key=lambda x: x.index))

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def channel_index_grouped_sort(
    channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo
) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign single channel per chart. Channels are grouped by the same index and
    sorted by type.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, D2, C0, C2, M0, M2, A0, A2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    chan_type_dict = defaultdict(list)
    inds = set()

    for chan in channels:
        chan_type_dict[type(chan)].append(chan)
        inds.add(chan.index)

    d_chans = chan_type_dict.get(pulse.DriveChannel, [])
    d_chans = sorted(d_chans, key=lambda x: x.index, reverse=True)

    u_chans = chan_type_dict.get(pulse.ControlChannel, [])
    u_chans = sorted(u_chans, key=lambda x: x.index, reverse=True)

    m_chans = chan_type_dict.get(pulse.MeasureChannel, [])
    m_chans = sorted(m_chans, key=lambda x: x.index, reverse=True)

    a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
    a_chans = sorted(a_chans, key=lambda x: x.index, reverse=True)

    ordered_channels = []

    for ind in sorted(inds):
        # drive channel
        if len(d_chans) > 0 and d_chans[-1].index == ind:
            ordered_channels.append(d_chans.pop())
        # control channel
        if len(u_chans) > 0 and u_chans[-1].index == ind:
            ordered_channels.append(u_chans.pop())
        # measure channel
        if len(m_chans) > 0 and m_chans[-1].index == ind:
            ordered_channels.append(m_chans.pop())
        # acquire channel
        if formatter["control.show_acquire_channel"]:
            if len(a_chans) > 0 and a_chans[-1].index == ind:
                ordered_channels.append(a_chans.pop())

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def channel_index_grouped_sort_u(
    channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo
) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign single channel per chart. Channels are grouped by the same index and
    sorted by type except for control channels. Control channels are added to the
    end of other channels.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, D2, C0, C2, M0, M2, A0, A2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    chan_type_dict = defaultdict(list)
    inds = set()

    for chan in channels:
        chan_type_dict[type(chan)].append(chan)
        inds.add(chan.index)

    d_chans = chan_type_dict.get(pulse.DriveChannel, [])
    d_chans = sorted(d_chans, key=lambda x: x.index, reverse=True)

    m_chans = chan_type_dict.get(pulse.MeasureChannel, [])
    m_chans = sorted(m_chans, key=lambda x: x.index, reverse=True)

    a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
    a_chans = sorted(a_chans, key=lambda x: x.index, reverse=True)

    u_chans = chan_type_dict.get(pulse.ControlChannel, [])
    u_chans = sorted(u_chans, key=lambda x: x.index)

    ordered_channels = []

    for ind in sorted(inds):
        # drive channel
        if len(d_chans) > 0 and d_chans[-1].index == ind:
            ordered_channels.append(d_chans.pop())
        # measure channel
        if len(m_chans) > 0 and m_chans[-1].index == ind:
            ordered_channels.append(m_chans.pop())
        # acquire channel
        if formatter["control.show_acquire_channel"]:
            if len(a_chans) > 0 and a_chans[-1].index == ind:
                ordered_channels.append(a_chans.pop())

    # control channels
    ordered_channels.extend(u_chans)

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def qubit_index_sort(
    channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo
) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign multiple channels per chart. Channels associated with the same qubit
    are grouped in the same chart and sorted by qubit index in ascending order.

    Acquire channels are not shown.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [Q0, Q1, Q2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    _removed = (
        pulse.channels.AcquireChannel,
        pulse.channels.MemorySlot,
        pulse.channels.RegisterSlot,
    )

    qubit_channel_map = defaultdict(list)

    for chan in channels:
        if isinstance(chan, _removed):
            continue
        qubit_channel_map[device.get_qubit_index(chan)].append(chan)

    sorted_map = sorted(qubit_channel_map.items(), key=lambda x: x[0])

    for qind, chans in sorted_map:
        yield f"Q{qind:d}", chans


def time_map_in_ns(
    time_window: Tuple[int, int], axis_breaks: List[Tuple[int, int]], dt: Optional[float] = None
) -> types.HorizontalAxis:
    """Layout function for the horizontal axis formatting.

    Calculate axis break and map true time to axis labels. Generate equispaced
    6 horizontal axis ticks. Convert into seconds if ``dt`` is provided.

    Args:
        time_window: Left and right edge of this graph.
        axis_breaks: List of axis break period.
        dt: Time resolution of system.

    Returns:
        Axis formatter object.
    """
    # shift time axis
    t0, t1 = time_window
    t0_shift = t0
    t1_shift = t1

    axis_break_pos = []
    offset_accumulation = 0
    for t0b, t1b in axis_breaks:
        if t1b < t0 or t0b > t1:
            continue
        if t0 > t1b:
            t0_shift -= t1b - t0b
        if t1 > t1b:
            t1_shift -= t1b - t0b
        axis_break_pos.append(t0b - offset_accumulation)
        offset_accumulation += t1b - t0b

    # axis label
    axis_loc = np.linspace(max(t0_shift, 0), t1_shift, 6)
    axis_label = axis_loc.copy()

    for t0b, t1b in axis_breaks:
        offset = t1b - t0b
        axis_label = np.where(axis_label > t0b, axis_label + offset, axis_label)

    # consider time resolution
    if dt:
        label = "Time (ns)"
        axis_label *= dt * 1e9
    else:
        label = "System cycle time (dt)"

    formatted_label = [f"{val:.0f}" for val in axis_label]

    return types.HorizontalAxis(
        window=(t0_shift, t1_shift),
        axis_map=dict(zip(axis_loc, formatted_label)),
        axis_break_pos=axis_break_pos,
        label=label,
    )


def detail_title(program: Union[pulse.Waveform, pulse.Schedule], device: DrawerBackendInfo) -> str:
    """Layout function for generating figure title.

    This layout writes program name, program duration, and backend name in the title.
    """
    title_str = []

    # add program name
    title_str.append(f"Name: {program.name}")

    # add program duration
    dt = device.dt * 1e9 if device.dt else 1.0
    title_str.append(f"Duration: {program.duration * dt:.1f} {'ns' if device.dt else 'dt'}")

    # add device name
    if device.backend_name != "no-backend":
        title_str.append(f"Backend: {device.backend_name}")

    return ", ".join(title_str)


def empty_title(program: Union[pulse.Waveform, pulse.Schedule], device: DrawerBackendInfo) -> str:
    """Layout function for generating an empty figure title."""
    return ""
