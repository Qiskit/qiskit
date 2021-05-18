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

"""A collection of backend information formatted to generate drawing data.

This instance will be provided to generator functions. The module provides an abstract
class :py:class:``DrawerBackendInfo`` with necessary methods to generate drawing objects.

Because the data structure of backend class may depend on providers, this abstract class
has an abstract factory method `create_from_backend`. Each subclass should provide
the factory method which conforms to the associated provider. By default we provide
:py:class:``OpenPulseBackendInfo`` class that has the factory method taking backends
satisfying OpenPulse specification [1].

This class can be also initialized without the factory method by manually specifying
required information. This may be convenient for visualizing a pulse program for simulator
backend that only has a device Hamiltonian information. This requires two mapping objects
for channel/qubit and channel/frequency along with the system cycle time.

If those information are not provided, this class will be initialized with a set of
empty data and the drawer illustrates a pulse program without any specific information.

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

    def __init__(
        self,
        name: Optional[str] = None,
        dt: Optional[float] = None,
        channel_frequency_map: Optional[Dict[pulse.channels.Channel, float]] = None,
        qubit_channel_map: Optional[Dict[int, List[pulse.channels.Channel]]] = None,
    ):
        """Create new backend information.

        Args:
            name: Name of the backend.
            dt: System cycle time.
            channel_frequency_map: Mapping of channel and associated frequency.
            qubit_channel_map: Mapping of qubit and associated channels.
        """
        self.backend_name = name or "no-backend"
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

        Returns:
            OpenPulseBackendInfo: New configured instance.
        """
        configuration = backend.configuration()
        defaults = backend.defaults()

        # load name
        name = backend.name()

        # load cycle time
        dt = configuration.dt

        # load frequencies
        chan_freqs = dict()

        chan_freqs.update(
            {pulse.DriveChannel(qind): freq for qind, freq in enumerate(defaults.qubit_freq_est)}
        )
        chan_freqs.update(
            {pulse.MeasureChannel(qind): freq for qind, freq in enumerate(defaults.meas_freq_est)}
        )
        for qind, u_lo_mappers in enumerate(configuration.u_channel_lo):
            temp_val = 0.0 + 0.0j
            for u_lo_mapper in u_lo_mappers:
                temp_val += defaults.qubit_freq_est[u_lo_mapper.q] * u_lo_mapper.scale
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

        return OpenPulseBackendInfo(
            name=name, dt=dt, channel_frequency_map=chan_freqs, qubit_channel_map=qubit_channel_map
        )
