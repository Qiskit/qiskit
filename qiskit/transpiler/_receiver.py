# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Receiver module for holding objects that take call backs.
"""

from collections import OrderedDict
from qiskit._qiskiterror import QISKitError


class Reciever(object):
    """A receiver class that holds instances of objects
    (such as) progressbars that recieve call back info.
    """
    def __init__(self):
        self._channels = OrderedDict()
        self.channel_id = 1

    def get_channels(self):
        """Getter to grab available channels.
        """
        return self._channels

    def add_channel(self, transmitter):
        """Add channel to the recievers channels.

        Parameters:
            transmitter (object): Object to be added to channels.

        Returns:
            int: Id of added channel.
        """
        self._channels[self.channel_id] = transmitter
        _channel_id = self.channel_id
        self.channel_id += 1
        return _channel_id

    def remove_channel(self, index):
        """Remove a channel from the receiver by index number.

        Parameters:
            index (int): Index fo channel to remove.

        Raises:
            QISKitError: Index not in receiver keys.
        """
        if index in self._channels.keys():
            del self._channels[index]
        else:
            raise QISKitError('Index not in receiver channels.')

    channels = property(get_channels, add_channel)


receiver = Reciever()  # pylint: disable=C0103
