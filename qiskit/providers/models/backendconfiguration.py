# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Backend Configuration Classes."""
import re
import copy
import warnings
from typing import Dict, List, Any, Iterable, Union
from collections import defaultdict

from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (Channel, DriveChannel, MeasureChannel,
                                   ControlChannel, AcquireChannel)


class GateConfig:
    """Class representing a Gate Configuration

    Attributes:
        name: the gate name as it will be referred to in Qasm.
        parameters: variable names for the gate parameters (if any).
        qasm_def: definition of this gate in terms of Qasm primitives U
                  and CX.
    """

    def __init__(self, name, parameters, qasm_def, coupling_map=None,
                 latency_map=None, conditional=None, description=None):
        """Initialize a GateConfig object

        Args:
            name (str): the gate name as it will be referred to in Qasm.
            parameters (list): variable names for the gate parameters (if any)
                               as a list of strings.
            qasm_def (str): definition of this gate in terms of Qasm primitives
                            U and CX.
            coupling_map (list): An optional coupling map for the gate. In
                the form of a list of lists of integers representing the qubit
                groupings which are coupled by this gate.
            latency_map (list): An optional map of latency for the gate. In the
                the form of a list of lists of integers of either 0 or 1
                representing an array of dimension
                len(coupling_map) X n_registers that specifies the register
                latency (1: fast, 0: slow) conditional operations on the gate
            conditional (bool): Optionally specify whether this gate supports
                conditional operations (true/false). If this is not specified,
                then the gate inherits the conditional property of the backend.
            description (str): Description of the gate operation
        """

        self.name = name
        self.parameters = parameters
        self.qasm_def = qasm_def
        # coupling_map with length 0 is invalid
        if coupling_map:
            self.coupling_map = coupling_map
        # latency_map with length 0 is invalid
        if latency_map:
            self.latency_map = latency_map
        if conditional is not None:
            self.conditional = conditional
        if description is not None:
            self.description = description

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = {
            'name': self.name,
            'parameters': self.parameters,
            'qasm_def': self.qasm_def,
        }
        if hasattr(self, 'coupling_map'):
            out_dict['coupling_map'] = self.coupling_map
        if hasattr(self, 'latency_map'):
            out_dict['latency_map'] = self.latency_map
        if hasattr(self, 'conditional'):
            out_dict['conditional'] = self.conditional
        if hasattr(self, 'description'):
            out_dict['description'] = self.description
        return out_dict

    def __eq__(self, other):
        if isinstance(other, GateConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        out_str = "GateConfig(%s, %s, %s" % (self.name, self.parameters,
                                             self.qasm_def)
        for i in ['coupling_map', 'latency_map', 'conditional', 'description']:
            if hasattr(self, i):
                out_str += ', ' + repr(getattr(self, i))
        out_str += ')'
        return out_str


class UchannelLO:
    """Class representing a U Channel LO

    Attributes:
        q: Qubit that scale corresponds too.
        scale: Scale factor for qubit frequency.
    """

    def __init__(self, q, scale):
        """Initialize a UchannelLOSchema object

        Args:
            q (int): Qubit that scale corresponds too. Must be >= 0.
            scale (complex): Scale factor for qubit frequency.

        Raises:
            QiskitError: If q is < 0
        """
        if q < 0:
            raise QiskitError('q must be >=0')
        self.q = q
        self.scale = scale

    @classmethod
    def from_dict(cls, data):
        """Create a new UchannelLO object from a dictionary.

        Args:
            data (dict): A dictionary representing the UChannelLO to
                create. It will be in the same format as output by
                :func:`to_dict`.

        Returns:
            UchannelLO: The UchannelLO from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the UChannelLO.

        Returns:
            dict: The dictionary form of the UChannelLO.
        """
        out_dict = {
            'q': self.q,
            'scale': self.scale,
        }
        return out_dict

    def __eq__(self, other):
        if isinstance(other, UchannelLO):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        return "UchannelLO(%s, %s)" % (self.q, self.scale)


class QasmBackendConfiguration:
    """Class representing a Qasm Backend Configuration.

    Attributes:
        backend_name: backend name.
        backend_version: backend version in the form X.Y.Z.
        n_qubits: number of qubits.
        basis_gates: list of basis gates names on the backend.
        gates: list of basis gates on the backend.
        local: backend is local or remote.
        simulator: backend is a simulator.
        conditional: backend supports conditional operations.
        open_pulse: backend supports open pulse.
        memory: backend supports memory.
        max_shots: maximum number of shots supported.
    """

    _data = {}

    def __init__(self, backend_name, backend_version, n_qubits,
                 basis_gates, gates, local, simulator,
                 conditional, open_pulse, memory,
                 max_shots, coupling_map, max_experiments=None,
                 sample_name=None, n_registers=None, register_map=None,
                 configurable=None, credits_required=None, online_date=None,
                 display_name=None, description=None, tags=None, **kwargs):
        """Initialize a QasmBackendConfiguration Object

        Args:
            backend_name (str): The backend name
            backend_version (str): The backend version in the form X.Y.Z
            n_qubits (int): the number of qubits for the backend
            basis_gates (list): The list of strings for the basis gates of the
                backends
            gates (list): The list of GateConfig objects for the basis gates of
                the backend
            local (bool): True if the backend is local or False if remote
            simulator (bool): True if the backend is a simulator
            conditional (bool): True if the backend supports conditional
                operations
            open_pulse (bool): True if the backend supports OpenPulse
            memory (bool): True if the backend supports memory
            max_shots (int): The maximum number of shots allowed on the backend
            coupling_map (list): The coupling map for the device
            max_experiments (int): The maximum number of experiments per job
            sample_name (str): Sample name for the backend
            n_registers (int): Number of register slots available for feedback
                (if conditional is True)
            register_map (list): An array of dimension n_qubits X
                n_registers that specifies whether a qubit can store a
                measurement in a certain register slot.
            configurable (bool): True if the backend is configurable, if the
                backend is a simulator
            credits_required (bool): True if backend requires credits to run a
                job.
            online_date (datetime): The date that the device went online
            display_name (str): Alternate name field for the backend
            description (str): A description for the backend
            tags (list): A list of string tags to describe the backend
            **kwargs: optional fields
        """
        self._data = {}

        self.backend_name = backend_name
        self.backend_version = backend_version
        self.n_qubits = n_qubits
        self.basis_gates = basis_gates
        self.gates = gates
        self.local = local
        self.simulator = simulator
        self.conditional = conditional
        self.open_pulse = open_pulse
        self.memory = memory
        self.max_shots = max_shots
        self.coupling_map = coupling_map
        # max_experiments must be >=1
        if max_experiments:
            self.max_experiments = max_experiments
        if sample_name is not None:
            self.sample_name = sample_name
        # n_registers must be >=1
        if n_registers:
            self.n_registers = 1
        # register_map must have at least 1 entry
        if register_map:
            self.register_map = register_map
        if configurable is not None:
            self.configurable = configurable
        if credits_required is not None:
            self.credits_required = credits_required
        if online_date is not None:
            self.online_date = online_date
        if display_name is not None:
            self.display_name = display_name
        if description is not None:
            self.description = description
        if tags is not None:
            self.tags = tags

        # Add pulse properties here becuase some backends do not
        # fit within the Qasm / Pulse backend partitioning in Qiskit
        if 'dt' in kwargs.keys():
            kwargs['dt'] *= 1e-9
        if 'dtm' in kwargs.keys():
            kwargs['dtm'] *= 1e-9

        if 'qubit_lo_range' in kwargs.keys():
            kwargs['qubit_lo_range'] = [[min_range * 1e9, max_range * 1e9] for
                                        (min_range, max_range) in kwargs['qubit_lo_range']]

        if 'meas_lo_range' in kwargs.keys():
            kwargs['meas_lo_range'] = [[min_range * 1e9, max_range * 1e9] for
                                       (min_range, max_range) in kwargs['meas_lo_range']]

        # convert rep_times from μs to sec
        if 'rep_times' in kwargs.keys():
            kwargs['rep_times'] = [_rt * 1e-6 for _rt in kwargs['rep_times']]

        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError('Attribute %s is not defined' % name)

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                         It will be in the same format as output by
                         :func:`to_dict`.
        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop('gates')]
        in_data['gates'] = gates
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = {
            'backend_name': self.backend_name,
            'backend_version': self.backend_version,
            'n_qubits': self.n_qubits,
            'basis_gates': self.basis_gates,
            'gates': [x.to_dict() for x in self.gates],
            'local': self.local,
            'simulator': self.simulator,
            'conditional': self.conditional,
            'open_pulse': self.open_pulse,
            'memory': self.memory,
            'max_shots': self.max_shots,
            'coupling_map': self.coupling_map,
        }
        for kwarg in ['max_experiments', 'sample_name', 'n_registers',
                      'register_map', 'configurable', 'credits_required',
                      'online_date', 'display_name', 'description',
                      'tags']:
            if hasattr(self, kwarg):
                out_dict[kwarg] = getattr(self, kwarg)

        out_dict.update(self._data)

        if 'dt' in out_dict:
            out_dict['dt'] *= 1e-9
        if 'dtm' in out_dict:
            out_dict['dtm'] *= 1e-9

        if 'qubit_lo_range' in out_dict:
            out_dict['qubit_lo_range'] = [
                [min_range * 1e9, max_range * 1e9] for
                (min_range, max_range) in out_dict['qubit_lo_range']
            ]

        if 'meas_lo_range' in out_dict:
            out_dict['meas_lo_range'] = [
                [min_range * 1e9, max_range * 1e9] for
                (min_range, max_range) in out_dict['meas_lo_range']
            ]

        # convert rep_times from μs to sec
        if 'rep_times' in out_dict:
            out_dict['rep_times'] = [_rt * 1e-6 for _rt in out_dict['rep_times']]

        return out_dict

    @property
    def num_qubits(self):
        """Returns the number of qubits.

        In future, `n_qubits` should be replaced in favor of `num_qubits` for consistent use
        throughout Qiskit. Until this is properly refactored, this property serves as intermediate
        solution.
        """
        return self.n_qubits

    def __eq__(self, other):
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __contains__(self, item):
        return item in self.__dict__


class BackendConfiguration(QasmBackendConfiguration):
    """Backwards compat shim representing an abstract backend configuration."""
    pass


class PulseBackendConfiguration(QasmBackendConfiguration):
    """Static configuration state for an OpenPulse enabled backend. This contains information
    about the set up of the device which can be useful for building Pulse programs.
    """

    def __init__(self,
                 backend_name: str,
                 backend_version: str,
                 n_qubits: int,
                 basis_gates: List[str],
                 gates: GateConfig,
                 local: bool,
                 simulator: bool,
                 conditional: bool,
                 open_pulse: bool,
                 memory: bool,
                 max_shots: int,
                 coupling_map,
                 n_uchannels: int,
                 u_channel_lo: List[List[UchannelLO]],
                 meas_levels: List[int],
                 qubit_lo_range: List[List[float]],
                 meas_lo_range: List[List[float]],
                 dt: float,
                 dtm: float,
                 rep_times: List[float],
                 meas_kernels: List[str],
                 discriminators: List[str],
                 dynamic_reprate_enabled: bool = False,
                 rep_delay_range: List[float] = None,
                 default_rep_delay: float = None,
                 hamiltonian: Dict[str, Any] = None,
                 channel_bandwidth=None,
                 acquisition_latency=None,
                 conditional_latency=None,
                 meas_map=None,
                 max_experiments=None,
                 sample_name=None,
                 n_registers=None,
                 register_map=None,
                 configurable=None,
                 credits_required=None,
                 online_date=None,
                 display_name=None,
                 description=None,
                 tags=None,
                 channels: Dict[str, Any] = None,
                 **kwargs):
        """
        Initialize a backend configuration that contains all the extra configuration that is made
        available for OpenPulse backends.

        Args:
            backend_name: backend name.
            backend_version: backend version in the form X.Y.Z.
            n_qubits: number of qubits.
            basis_gates: list of basis gates names on the backend.
            gates: list of basis gates on the backend.
            local: backend is local or remote.
            simulator: backend is a simulator.
            conditional: backend supports conditional operations.
            open_pulse: backend supports open pulse.
            memory: backend supports memory.
            max_shots: maximum number of shots supported.
            coupling_map (list): The coupling map for the device
            n_uchannels: Number of u-channels.
            u_channel_lo: U-channel relationship on device los.
            meas_levels: Supported measurement levels.
            qubit_lo_range: Qubit lo ranges for each qubit with form (min, max) in GHz.
            meas_lo_range: Measurement lo ranges for each qubit with form (min, max) in GHz.
            dt: Qubit drive channel timestep in nanoseconds.
            dtm: Measurement drive channel timestep in nanoseconds.
            rep_times: Supported repetition times (program execution time) for backend in μs.
            meas_kernels: Supported measurement kernels.
            discriminators: Supported discriminators.
            dynamic_reprate_enabled: whether delay between programs can be set dynamically
                (ie via ``rep_delay``). Defaults to False.
            rep_delay_range: 2d list defining supported range of repetition delays (delay
                programs) for backend in μs. First entry is lower end of the range, second entry is
                higher end of the range. Optional, but will be specified when
                ``dynamic_reprate_enabled=True``.
            default_rep_delay: Value of ``rep_delay`` if not specified by user and
                ``dynamic_reprate_enabled=True``.
            hamiltonian: An optional dictionary with fields characterizing the system hamiltonian.
            channel_bandwidth (list): Bandwidth of all channels
                (qubit, measurement, and U)
            acquisition_latency (list): Array of dimension
                n_qubits x n_registers. Latency (in units of dt) to write a
                measurement result from qubit n into register slot m.
            conditional_latency (list): Array of dimension n_channels
                [d->u->m] x n_registers. Latency (in units of dt) to do a
                conditional operation on channel n from register slot m
            meas_map (list): Grouping of measurement which are multiplexed
            max_experiments (int): The maximum number of experiments per job
            sample_name (str): Sample name for the backend
            n_registers (int): Number of register slots available for feedback
                (if conditional is True)
            register_map (list): An array of dimension n_qubits X
                n_registers that specifies whether a qubit can store a
                measurement in a certain register slot.
            configurable (bool): True if the backend is configurable, if the
                backend is a simulator
            credits_required (bool): True if backend requires credits to run a
                job.
            online_date (datetime): The date that the device went online
            display_name (str): Alternate name field for the backend
            description (str): A description for the backend
            tags (list): A list of string tags to describe the backend
            channels: An optional dictionary containing information of each channel -- their
                purpose, type, and qubits operated on.
            **kwargs: Optional fields.
        """
        self.n_uchannels = n_uchannels
        self.u_channel_lo = u_channel_lo
        self.meas_levels = meas_levels
        self.qubit_lo_range = [[min_range * 1e9, max_range * 1e9] for
                               (min_range, max_range) in qubit_lo_range]
        self.meas_lo_range = [[min_range * 1e9, max_range * 1e9] for
                              (min_range, max_range) in meas_lo_range]
        self.meas_kernels = meas_kernels
        self.discriminators = discriminators
        self.hamiltonian = hamiltonian

        self.dynamic_reprate_enabled = dynamic_reprate_enabled

        self.rep_times = [_rt * 1e-6 for _rt in rep_times]  # convert to sec
        if rep_delay_range:
            self.rep_delay_range = [_rd * 1e-6 for _rd in rep_delay_range]  # convert to sec
        if default_rep_delay:
            self.default_rep_delay = default_rep_delay * 1e-6   # convert to sec

        self.dt = dt * 1e-9  # pylint: disable=invalid-name
        self.dtm = dtm * 1e-9

        if channels is not None:
            self.channels = channels

            (self._qubit_channel_map,
             self._channel_qubit_map,
             self._control_channels) = self._parse_channels(channels=channels)

        if channel_bandwidth is not None:
            self.channel_bandwidth = [[min_range * 1e9, max_range * 1e9] for
                                      (min_range, max_range) in channel_bandwidth]
        if acquisition_latency is not None:
            self.acquisition_latency = acquisition_latency
        if conditional_latency is not None:
            self.conditional_latency = conditional_latency
        if meas_map is not None:
            self.meas_map = meas_map
        super().__init__(backend_name=backend_name, backend_version=backend_version,
                         n_qubits=n_qubits, basis_gates=basis_gates, gates=gates,
                         local=local, simulator=simulator, conditional=conditional,
                         open_pulse=open_pulse, memory=memory, max_shots=max_shots,
                         coupling_map=coupling_map, max_experiments=max_experiments,
                         sample_name=sample_name, n_registers=n_registers,
                         register_map=register_map, configurable=configurable,
                         credits_required=credits_required, online_date=online_date,
                         display_name=display_name, description=description,
                         tags=tags, **kwargs)

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                It will be in the same format as output by :func:`to_dict`.

        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop('gates')]
        in_data['gates'] = gates
        input_uchannels = in_data.pop('u_channel_lo')
        u_channels = []
        for channel in input_uchannels:
            u_channels.append([UchannelLO.from_dict(x) for x in channel])
        in_data['u_channel_lo'] = u_channels
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = super().to_dict()
        u_channel_lo = []
        for x in self.u_channel_lo:
            channel = []
            for y in x:
                channel.append(y.to_dict())
            u_channel_lo.append(channel)
        out_dict.update({
            'n_uchannels': self.n_uchannels,
            'u_channel_lo': u_channel_lo,
            'meas_levels': self.meas_levels,
            'qubit_lo_range': self.qubit_lo_range,
            'meas_lo_range': self.meas_lo_range,
            'meas_kernels': self.meas_kernels,
            'discriminators': self.discriminators,
            'hamiltonian': self.hamiltonian,
            'rep_times': self.rep_times,
            'dt': self.dt,
            'dtm': self.dtm,
            'dynamic_reprate_enabled': self.dynamic_reprate_enabled
        })
        if hasattr(self, 'rep_delay_range'):
            out_dict['rep_delay_range'] = [_rd * 1e6 for _rd in self.rep_delay_range]
        if hasattr(self, 'default_rep_delay'):
            out_dict['default_rep_delay'] = self.default_rep_delay*1e6
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = self.channel_bandwidth
        if hasattr(self, 'meas_map'):
            out_dict['meas_map'] = self.meas_map
        if hasattr(self, 'acquisition_latency'):
            out_dict['acquisition_latency'] = self.acquisition_latency
        if hasattr(self, 'conditional_latency'):
            out_dict['conditional_latency'] = self.conditional_latency
        if 'channels' in out_dict:
            out_dict.pop('_qubit_channel_map')
            out_dict.pop('_channel_qubit_map')
            out_dict.pop('_control_channels')

        if self.qubit_lo_range:
            out_dict['qubit_lo_range'] = [
                [min_range * 1e-9, max_range * 1e-9] for
                (min_range, max_range) in self.qubit_lo_range]

        if self.meas_lo_range:
            out_dict['meas_lo_range'] = [
                [min_range * 1e-9, max_range * 1e-9] for
                (min_range, max_range) in self.meas_lo_range]

        if self.rep_times:
            out_dict['rep_times'] = [_rt * 1e6 for _rt in self.rep_times]

        out_dict['dt'] = out_dict['dt'] * 1e9  # pylint: disable=invalid-name
        out_dict['dtm'] = out_dict['dtm'] * 1e9

        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = [
                [min_range * 1e-9, max_range * 1e-9] for
                (min_range, max_range) in self.channel_bandwidth]

        return out_dict

    def __eq__(self, other):
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    @property
    def sample_rate(self) -> float:
        """Sample rate of the signal channels in Hz (1/dt)."""
        return 1.0 / self.dt

    def drive(self, qubit: int) -> DriveChannel:
        """
        Return the drive channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.

        Returns:
            Qubit drive channel.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError("Invalid index for {}-qubit system.".format(qubit))
        return DriveChannel(qubit)

    def measure(self, qubit: int) -> MeasureChannel:
        """
        Return the measure stimulus channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.
        Returns:
            Qubit measurement stimulus line.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError("Invalid index for {}-qubit system.".format(qubit))
        return MeasureChannel(qubit)

    def acquire(self, qubit: int) -> AcquireChannel:
        """
        Return the acquisition channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.
        Returns:
            Qubit measurement acquisition line.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError("Invalid index for {}-qubit systems.".format(qubit))
        return AcquireChannel(qubit)

    def control(self, qubits: Iterable[int] = None,
                channel: int = None) -> List[ControlChannel]:
        """
        Return the secondary drive channel for the given qubit -- typically utilized for
        controlling multiqubit interactions. This channel is derived from other channels.

        Args:
            qubits: Tuple or list of qubits of the form `(control_qubit, target_qubit)`.
            channel: Deprecated.

        Raises:
            BackendConfigurationError: If the ``qubits`` is not a part of the system or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of control channels.
        """
        if channel is not None:
            warnings.warn('The channel argument has been deprecated in favor of qubits. '
                          'This method will now return accurate ControlChannels determined '
                          'by qubit indices.',
                          DeprecationWarning)
            qubits = [channel]
        try:
            if isinstance(qubits, list):
                qubits = tuple(qubits)
            return self._control_channels[qubits]
        except KeyError:
            raise BackendConfigurationError("Couldn't find the ControlChannel operating on qubits "
                                            "{} on {}-qubit system. The ControlChannel information"
                                            " is retrieved from the "
                                            " backend.".format(qubits, self.n_qubits))
        except AttributeError:
            raise BackendConfigurationError("This backend - '{}' does not provide channel "
                                            "information.".format(self.backend_name))

    def get_channel_qubits(self, channel: Channel) -> List[int]:
        """
        Return a list of indices for qubits which are operated on directly by the given ``channel``.

        Raises:
            BackendConfigurationError: If ``channel`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of qubits operated on my the given ``channel``.
        """
        try:
            return self._channel_qubit_map[channel]
        except KeyError:
            raise BackendConfigurationError("Couldn't find the Channel - {}".format(channel))
        except AttributeError:
            raise BackendConfigurationError("This backend - '{}' does not provide channel "
                                            "information.".format(self.backend_name))

    def get_qubit_channels(self, qubit: Union[int, Iterable[int]]) -> List[Channel]:
        r"""Return a list of channels which operate on the given ``qubit``.

        Raises:
            BackendConfigurationError: If ``qubit`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of ``Channel``\s operated on my the given ``qubit``.
        """
        channels = set()
        try:
            if isinstance(qubit, int):
                for key in self._qubit_channel_map.keys():
                    if qubit in key:
                        channels.update(self._qubit_channel_map[key])
                if len(channels) == 0:
                    raise KeyError
            elif isinstance(qubit, list):
                qubit = tuple(qubit)
                channels.update(self._qubit_channel_map[qubit])
            elif isinstance(qubit, tuple):
                channels.update(self._qubit_channel_map[qubit])
            return list(channels)
        except KeyError:
            raise BackendConfigurationError("Couldn't find the qubit - {}".format(qubit))
        except AttributeError:
            raise BackendConfigurationError("This backend - '{}' does not provide channel "
                                            "information.".format(self.backend_name))

    def describe(self, channel: ControlChannel) -> Dict[DriveChannel, complex]:
        """
        Return a basic description of the channel dependency. Derived channels are given weights
        which describe how their frames are linked to other frames.
        For instance, the backend could be configured with this setting::

            u_channel_lo = [
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]
            ]

        Then, this method can be used as follows::

            backend.configuration().describe(ControlChannel(1))
            >>> {DriveChannel(0): -1, DriveChannel(1): 1}

        Args:
            channel: The derived channel to describe.
        Raises:
            BackendConfigurationError: If channel is not a ControlChannel.
        Returns:
            Control channel derivations.
        """
        if not isinstance(channel, ControlChannel):
            raise BackendConfigurationError("Can only describe ControlChannels.")
        result = {}
        for u_chan_lo in self.u_channel_lo[channel.index]:
            result[DriveChannel(u_chan_lo.q)] = u_chan_lo.scale
        return result

    def _parse_channels(self, channels: Dict[set, Any]) -> Dict[Any, Any]:
        r"""
        Generates a dictionaries of ``Channel``\s, and tuple of qubit(s) they operate on.

        Args:
            channels: An optional dictionary containing information of each channel -- their
                purpose, type, and qubits operated on.

        Returns:
            qubit_channel_map: Dictionary mapping tuple of qubit(s) to list of ``Channel``\s.
            channel_qubit_map: Dictionary mapping ``Channel`` to list of qubit(s).
            control_channels: Dictionary mapping tuple of qubit(s), to list of
                ``ControlChannel``\s.
        """
        qubit_channel_map = defaultdict(list)
        channel_qubit_map = defaultdict(list)
        control_channels = defaultdict(list)
        channels_dict = {
            DriveChannel.prefix: DriveChannel,
            ControlChannel.prefix: ControlChannel,
            MeasureChannel.prefix: MeasureChannel,
            'acquire': AcquireChannel
        }
        for channel, config in channels.items():
            channel_prefix, index = self._get_channel_prefix_index(channel)
            channel_type = channels_dict[channel_prefix]
            qubits = tuple(config['operates']['qubits'])
            if channel_prefix in channels_dict:
                qubit_channel_map[qubits].append(channel_type(index))
                channel_qubit_map[(channel_type(index))].extend(list(qubits))
                if channel_prefix == ControlChannel.prefix:
                    control_channels[qubits].append(channel_type(index))
        return dict(qubit_channel_map), dict(channel_qubit_map), dict(control_channels)

    def _get_channel_prefix_index(self, channel: str) -> str:
        """Return channel prefix and index from the given ``channel``.

        Args:
            channel: Name of channel.

        Raises:
            BackendConfigurationError: If invalid channel name is found.

        Return:
            Channel name and index. For example, if ``channel=acquire0``, this method
            returns ``acquire`` and ``0``.
        """
        channel_prefix = re.match(r"(?P<channel>[a-z]+)(?P<index>[0-9]+)", channel)
        try:
            return channel_prefix.group('channel'), int(channel_prefix.group('index'))
        except AttributeError:
            raise BackendConfigurationError("Invalid channel name - '{}' found.".format(channel))
