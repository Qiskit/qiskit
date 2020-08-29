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
A collection of functions that decide the layout of a figure.

Currently this module provides functions to arrange the order of channels.

An end-user can write their own layouts with by providing a function with the signature:

    ```python
    def my_channel_layout(channels: List[Channel]) -> List[Channel]:
        ordered_channels = []
        # arrange order of channels

        return ordered_channels
    ```

The user-defined arrangement process can be assigned to the layout of the stylesheet:

    ```python
    my_custom_style = {
        'layout': {'channel': my_channel_layout}
    }
    ```

The user can set the custom stylesheet to the drawer interface.
"""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator

from qiskit import pulse
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo


def channel_type_grouped_sort(channels: List[pulse.channels.Channel],
                              formatter: Dict[str, Any],
                              device: DrawerBackendInfo) \
        -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Assign single channel per chart. Channels are grouped by type and
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
    a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
    ordered_channels.extend(sorted(a_chans, key=lambda x: x.index))

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def channel_index_grouped_sort(channels: List[pulse.channels.Channel],
                               formatter: Dict[str, Any],
                               device: DrawerBackendInfo) \
        -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Assign single channel per chart. Channels are grouped by the same index and
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

    for ind in inds:
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
        if len(a_chans) > 0 and a_chans[-1].index == ind:
            ordered_channels.append(a_chans.pop())

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def channel_index_grouped_sort_except_u(channels: List[pulse.channels.Channel],
                                        formatter: Dict[str, Any],
                                        device: DrawerBackendInfo) \
        -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Assign single channel per chart. Channels are grouped by the same index and
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

    for ind in inds:
        # drive channel
        if len(d_chans) > 0 and d_chans[-1].index == ind:
            ordered_channels.append(d_chans.pop())
        # measure channel
        if len(m_chans) > 0 and m_chans[-1].index == ind:
            ordered_channels.append(m_chans.pop())
        # acquire channel
        if len(a_chans) > 0 and a_chans[-1].index == ind:
            ordered_channels.append(a_chans.pop())

    # control channels
    ordered_channels.extend(u_chans)

    for chan in ordered_channels:
        yield chan.name.upper(), [chan]


def qubit_index_sort(channels: List[pulse.channels.Channel],
                     formatter: Dict[str, Any],
                     device: DrawerBackendInfo) \
        -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Assign multiple channels per chart. Channels associated with the same qubit
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
    qubit_channel_map = defaultdict(list)

    for chan in channels:
        if isinstance(chan, pulse.channels.AcquireChannel):
            continue
        qubit_channel_map[device.get_qubit_index(chan)].append(chan)

    sorted_map = sorted(qubit_channel_map.items(), key=lambda x: x[0])

    for qind, chans in sorted_map:
        yield 'Q{index:d}'.format(index=qind), chans
