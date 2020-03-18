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
import copy
from types import SimpleNamespace
from typing import Dict, List

from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel


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


class QasmBackendConfiguration(SimpleNamespace):
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
        self.__dict__.update(kwargs)

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        return self.from_dict(state)

    def __reduce__(self):
        return (self.__class__, (self.backend_name, self.backend_version,
                                 self.n_qubits, self.basis_gates, self.gates,
                                 self.local, self.simulator, self.conditional,
                                 self.open_pulse, self.memory, self.max_shots,
                                 self.coupling_map))

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
        out_dict = copy.copy(self.__dict__)
        out_dict['gates'] = [x.to_dict() for x in self.gates]
        return out_dict

    def __eq__(self, other):
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False


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
                 u_channel_lo: List[UchannelLO],
                 meas_levels: List[int],
                 qubit_lo_range: List[List[float]],
                 meas_lo_range: List[List[float]],
                 dt: float,
                 dtm: float,
                 rep_times: List[float],
                 meas_kernels: List[str],
                 discriminators: List[str],
                 hamiltonian: Dict[str, str] = None,
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
            rep_times: Supported repetition times for device in microseconds.
            meas_kernels: Supported measurement kernels.
            discriminators: Supported discriminators.
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

        self.rep_times = [_rt * 1e-6 for _rt in rep_times]
        self.dt = dt * 1e-9  # pylint: disable=invalid-name
        self.dtm = dtm * 1e-9

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
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        gates = [GateConfig.from_dict(x) for x in data.pop('gates')]
        data['gates'] = gates
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = super().to_dict()
        out_dict.update({
            'n_uchannels': self.n_uchannels,
            'u_channel_lo': self.u_channel_lo,
            'meas_levels': self.meas_levels,
            'qubit_lo_range': self.qubit_lo_range,
            'meas_lo_range': self.meas_lo_range,
            'meas_kernels': self.meas_kernels,
            'discriminators': self.discriminators,
            'hamiltonian': self.hamiltonian,
            'rep_times': self.rep_times,
            'dt': self.dt,
            'dtm': self.dtm,
        })
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = self.channel_bandwidth
        if hasattr(self, 'meas_map'):
            out_dict['meas_map'] = self.meas_map
        if hasattr(self, 'acquisition_latency'):
            out_dict['acquisition_latency'] = self.acquisition_latency
        if hasattr(self, 'conditional_latency'):
            out_dict['conditional_latency'] = self.conditional_latency
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

    def control(self, channel: int) -> ControlChannel:
        """
        Return the secondary drive channel for the given qubit -- typically utilized for
        controlling multiqubit interactions. This channel is derived from other channels.

        Returns:
            Qubit control channel.
        """
        # TODO: Determine this from the hamiltonian.
        return ControlChannel(channel)

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
