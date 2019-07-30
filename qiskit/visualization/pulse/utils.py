# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import re

from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel
from qiskit.visualization.exceptions import VisualizationError


def _get_channel_object(channels_str):
    """Get channel object instance from channel letter.

    Args:
        channels_str (list): List of channel letters.

    Returns:
        channel_to_plot: List of channel object instance corresponding to the input.

    Raises:
        VisualizationError: when invalid channel is specified.
    """
    defined_channels = [DriveChannel, MeasureChannel, ControlChannel, AcquireChannel]
    channel_format = re.compile(r"(?P<ch_name>[a-z]+)(?P<index>[0-9]+)")

    channels_to_plot = []
    for channel_str in channels_str:
        parsed = channel_format.match(channel_str)
        if parsed:
            ch_name = parsed.group('ch_name')
            for defined_channel in defined_channels:
                if defined_channel.prefix == ch_name:
                    ch_index = int(parsed.group('index'))
                    channels_to_plot.append(defined_channel(index=ch_index))
                    break
            else:
                raise VisualizationError('Input channel letter %s is not defined.' % ch_name)
        else:
            raise VisualizationError('Invalid channel %s is specified.' % channel_str)

    return channels_to_plot
