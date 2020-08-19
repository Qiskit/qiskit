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
from typing import List

from qiskit import pulse


# channel layout


def channel_type_grouped_sort(channels: List[pulse.channels.Channel]) \
        -> List[pulse.channels.Channel]:
    """Group channels into each type and arrange them in ascending order.

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, D2, C0, C2, M0, M2, A0, A2]

    Args:
        channels: Channels to show.

    Returns:
        Arranged channels.
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

    return ordered_channels


def channel_index_sort(channels: List[pulse.channels.Channel]) \
        -> List[pulse.channels.Channel]:
    """Arrange channels in ascending order.

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, C0, M0, A0, D2, C2, M2, A2]

    Args:
        channels: Channels to show.

    Returns:
        Arranged channels.
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

    return ordered_channels


def channel_index_sort_wo_control(channels: List[pulse.channels.Channel]) \
        -> List[pulse.channels.Channel]:
    """Arrange channels in ascending order, but control channels are grouped and
    added to the end of the list.

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, M0, A0, D2, M2, A2, C0, C2]

    Args:
        channels: Channels to show.

    Returns:
        Arranged channels.
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

    return ordered_channels
