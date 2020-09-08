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

# pylint: disable=invalid-name

"""A collection of formatted backend information to generate drawing data.

This module provide an abstract class :py:class:``DrawerBackendInfo`` with
necessary methods to generate drawing objects.

Because the data structure of backend class may depend on the provider,
this abstract class has a factory method `create_from_backend` as abstractmethod.

We provide :py:class:``OpenPulseBackendInfo`` class that has the
factory method that can handle backends that conform to OpenPulse specification [1].

On the other hand, this class can be initialized without the factory method,
i.e. a pulse program for simulator that only has device Hamiltonian information,
by manually specifying required information.
This requires two mappings for channel/qubit and channel/frequency and the system cycle time.

If information is not provided, this class is initialized as a collection of
empty information, and the drawer illustrates a pulse program so as to be agnostic to device.

Reference:
- [1] Qiskit Backend Specifications for OpenQASM and OpenPulse Experiments,
    https://arxiv.org/abs/1809.03452
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional

from qiskit import pulse
from qiskit.providers import BaseBackend, BackendConfigurationError


class DrawerBackendInfo(ABC):
    """Backend information to be used for the drawing data generation."""

    def __init__(self,
                 name: Optional[str] = None,
                 dt: Optional[float] = None,
                 channel_frequency_map: Optional[Dict[pulse.channels.Channel, float]] = None,
                 qubit_channel_map: Optional[Dict[int, List[pulse.channels.Channel]]] = None):
        """Create new backend information.

        Args:
            name: Name of the backend.
            dt: System cycle time.
            channel_frequency_map: Mapping of channel and associated frequency.
            qubit_channel_map: Mapping of qubit and associated channels.
        """
        self.backend_name = name or 'no-backend'
        self._dt = dt
        self._chan_freq_map = channel_frequency_map or dict()
        self._qubit_channel_map = qubit_channel_map or dict()

    @classmethod
    @abstractmethod
    def create_from_backend(cls, backend: BaseBackend):
        """Initialize a class with backend information provided by provider.

        Args:
            backend: Backend object.
        """
        raise NotImplementedError

    @property
    def dt(self):
        """Return cycle time."""
        return self._dt

    def get_qubit_index(self, chan: pulse.channels.Channel) -> Union[int, None]:
        """Get associated qubit index of given channel object."""
        for qind, chans in self._qubit_channel_map.items():
            if chan in chans:
                return qind
        else:
            return chan.index

    def get_channel_frequency(self, chan: pulse.channels.Channel) -> Union[float, None]:
        """Get frequency of given channel object."""
        return self._chan_freq_map.get(chan, None)


class OpenPulseBackendInfo(DrawerBackendInfo):
    """Drawing information of backend that conforms to OpenPulse specification."""

    @classmethod
    def create_from_backend(cls, backend: BaseBackend):
        """Initialize a class with backend information provided by provider.

        Args:
            backend: Backend object.
        """
        configuration = backend.configuration()
        defaults = backend.defaults()

        # load name
        name = backend.name()

        # load cycle time
        dt = configuration.dt

        # load frequencies
        chan_freqs = dict()

        chan_freqs.update({pulse.DriveChannel(qind): freq
                           for qind, freq in enumerate(defaults.qubit_freq_est)})
        chan_freqs.update({pulse.MeasureChannel(qind): freq
                           for qind, freq in enumerate(defaults.meas_freq_est)})
        for qind, u_lo_mappers in enumerate(configuration.u_channel_lo):
            temp_val = .0 + .0j
            for u_lo_mapper in u_lo_mappers:
                temp_val += defaults.qubit_freq_est[u_lo_mapper.q] * complex(*u_lo_mapper.scale)
            chan_freqs[pulse.ControlChannel(qind)] = temp_val.real

        # load qubit channel mapping
        qubit_channel_map = defaultdict(list)
        for qind in range(configuration.n_qubits):
            qubit_channel_map[qind].append(configuration.drive(qubit=qind))
            qubit_channel_map[qind].append(configuration.measure(qubit=qind))
            for tind in range(configuration.n_qubits):
                try:
                    qubit_channel_map[qind].extend(configuration.control(qubits=(qind, tind)))
                except BackendConfigurationError:
                    pass

        return OpenPulseBackendInfo(name=name,
                                    dt=dt,
                                    channel_frequency_map=chan_freqs,
                                    qubit_channel_map=qubit_channel_map)
