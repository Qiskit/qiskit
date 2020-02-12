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

"""Model and schema for backend configuration."""
from typing import Dict, List
import warnings

from marshmallow.validate import Length, OneOf, Range, Regexp

from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel
from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation import fields
from qiskit.validation.validate import PatternProperties
from qiskit.providers.exceptions import BackendConfigurationError


class GateConfigSchema(BaseSchema):
    """Schema for GateConfig."""

    # Required properties.
    name = fields.String(required=True)
    parameters = fields.List(fields.String(), required=True)
    qasm_def = fields.String(required=True)

    # Optional properties.
    coupling_map = fields.List(fields.List(fields.Integer(),
                                           validate=Length(min=1)),
                               validate=Length(min=1))
    latency_map = fields.List(fields.List(fields.Integer(validate=OneOf([0, 1])),
                                          validate=Length(min=1)),
                              validate=Length(min=1))
    conditional = fields.Boolean()
    description = fields.String()


class UchannelLOSchema(BaseSchema):
    """Schema for uchannel LO."""

    # Required properties.
    q = fields.Integer(required=True, validate=Range(min=0))
    scale = fields.Complex(required=True)


class PulseHamiltonianSchema(BaseSchema):
    """Schema for PulseHamiltonian."""

    # Required properties.
    h_str = fields.List(fields.String(), validate=Length(min=1), required=True)
    dim_osc = fields.List(fields.Integer(validate=Range(min=1)), required=True)
    dim_qub = fields.List(fields.Integer(validate=Range(min=2)), required=True)
    vars = fields.Dict(validate=PatternProperties({
        Regexp('^([a-z0-9])+$'): fields.InstructionParameter()
    }), required=True)


class BackendConfigurationSchema(BaseSchema):
    """Schema for BackendConfiguration."""
    # Required properties.
    backend_name = fields.String(required=True)
    backend_version = fields.String(required=True,
                                    validate=Regexp("[0-9]+.[0-9]+.[0-9]+$"))
    n_qubits = fields.Integer(required=True, validate=Range(min=1))
    basis_gates = fields.List(fields.String(), required=True)
    gates = fields.Nested(GateConfigSchema, required=True, many=True)
    local = fields.Boolean(required=True)
    simulator = fields.Boolean(required=True)
    conditional = fields.Boolean(required=True)
    open_pulse = fields.Boolean(required=True)
    memory = fields.Boolean(required=True)
    max_shots = fields.Integer(required=True, validate=Range(min=1))
    coupling_map = fields.List(fields.List(fields.Integer(), validate=Length(min=1)),
                               validate=Length(min=1), allow_none=True, required=True)

    # Optional properties.
    max_experiments = fields.Integer(validate=Range(min=1))
    sample_name = fields.String()
    n_registers = fields.Integer(validate=Range(min=1))
    register_map = fields.List(fields.List(fields.Integer(validate=OneOf([0, 1])),
                                           validate=Length(min=1)),
                               validate=Length(min=1))
    configurable = fields.Boolean()
    credits_required = fields.Boolean()
    online_date = fields.DateTime()
    display_name = fields.String()
    description = fields.String()
    tags = fields.List(fields.String())


class QasmBackendConfigurationSchema(BackendConfigurationSchema):
    """Schema for Qasm backend."""
    open_pulse = fields.Boolean(required=True, validate=OneOf([False]))


class PulseBackendConfigurationSchema(QasmBackendConfigurationSchema):
    """Schema for pulse backend"""
    # Required properties.
    open_pulse = fields.Boolean(required=True, validate=OneOf([True]))
    n_uchannels = fields.Integer(required=True, validate=Range(min=0))
    u_channel_lo = fields.List(fields.Nested(UchannelLOSchema, validate=Length(min=1),
                                             required=True, many=True))
    meas_levels = fields.List(fields.Integer(), validate=Length(min=1), required=True)
    qubit_lo_range = fields.List(fields.List(fields.Float(validate=Range(min=0)),
                                             validate=Length(equal=2)), required=True)
    meas_lo_range = fields.List(fields.List(fields.Float(validate=Range(min=0)),
                                            validate=Length(equal=2)), required=True)
    dt = fields.Float(required=True, validate=Range(min=0))  # pylint: disable=invalid-name
    dtm = fields.Float(required=True, validate=Range(min=0))
    rep_times = fields.List(fields.Integer(validate=Range(min=0)), required=True)
    meas_kernels = fields.List(fields.String(), required=True)
    discriminators = fields.List(fields.String(), required=True)

    # Optional properties.
    meas_map = fields.List(fields.List(fields.Integer(), validate=Length(min=1)))
    channel_bandwidth = fields.List(fields.List(fields.Float(), validate=Length(equal=2)))
    acquisition_latency = fields.List(fields.List(fields.Integer()))
    conditional_latency = fields.List(fields.List(fields.Integer()))
    hamiltonian = PulseHamiltonianSchema()


@bind_schema(GateConfigSchema)
class GateConfig(BaseModel):
    """Model for GateConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``GateConfigSchema``.

    Attributes:
        name: the gate name as it will be referred to in Qasm.
        parameters: variable names for the gate parameters (if any).
        qasm_def: definition of this gate in terms of Qasm primitives U
                  and CX.
    """

    def __init__(self, name: str, parameters: List[str], qasm_def: str, **kwargs):
        self.name = name
        self.parameters = parameters
        self.qasm_def = qasm_def

        super().__init__(**kwargs)


@bind_schema(UchannelLOSchema)
class UchannelLO(BaseModel):
    """Model for GateConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``GateConfigSchema``.

    Attributes:
        q: Qubit that scale corresponds too.
        scale: Scale factor for qubit frequency.
    """

    def __init__(self, q: int, scale: complex, **kwargs):
        self.q = q
        self.scale = scale

        super().__init__(q=q, scale=scale, **kwargs)


@bind_schema(BackendConfigurationSchema)
class BackendConfiguration(BaseModel):
    """Model for BackendConfiguration.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``BackendConfigurationSchema``.
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
        **kwargs: Optional fields.
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
                 **kwargs):
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

        super().__init__(**kwargs)


@bind_schema(QasmBackendConfigurationSchema)
class QasmBackendConfiguration(BackendConfiguration):
    """Model for QasmBackendConfiguration.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmBackendConfigurationSchema``.
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
        **kwargs: Optional fields.
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
                 **kwargs):
        super().__init__(backend_name=backend_name, backend_version=backend_version,
                         n_qubits=n_qubits, basis_gates=basis_gates, gates=gates,
                         local=local, simulator=simulator, conditional=conditional,
                         open_pulse=open_pulse, memory=memory, max_shots=max_shots,
                         **kwargs)


@bind_schema(PulseBackendConfigurationSchema)
class PulseBackendConfiguration(BackendConfiguration):
    """Static configuration state for an OpenPulse enabled backend. This contains information
    about the set up of the device which can be useful for building Pulse programs.
    """

    _dt_warning_done = False
    _rep_time_warning_done = False

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

        self._rep_times = [_rt * 1e-6 for _rt in rep_times]
        self._dt = dt * 1e-9
        self._dtm = dtm * 1e-9

        channel_bandwidth = kwargs.pop('channel_bandwidth', None)
        if channel_bandwidth:
            self.channel_bandwidth = [[min_range * 1e9, max_range * 1e9] for
                                      (min_range, max_range) in channel_bandwidth]

        super().__init__(backend_name=backend_name, backend_version=backend_version,
                         n_qubits=n_qubits, basis_gates=basis_gates, gates=gates,
                         local=local, simulator=simulator, conditional=conditional,
                         open_pulse=open_pulse, memory=memory, max_shots=max_shots,
                         **kwargs)

    @property
    def dt(self) -> float:  # pylint: disable=invalid-name
        """Drive channel sampling time in seconds(s)."""
        # only raise dt warning once
        if not PulseBackendConfiguration._dt_warning_done:
            warnings.warn('`dt` and `dtm` now have units of seconds(s) rather '
                          'than nanoseconds(ns).')
            PulseBackendConfiguration._dt_warning_done = True

        return self._dt

    @property
    def dtm(self) -> float:  # pylint: disable=invalid-name
        """Measure channel sampling time in seconds(s)."""
        # only raise dt warning once
        if not PulseBackendConfiguration._dt_warning_done:
            warnings.warn('`dt` and `dtm` now have units of seconds(s) rather '
                          'than nanoseconds(ns).')
            PulseBackendConfiguration._dt_warning_done = True

        return self._dtm

    @property
    def rep_times(self) -> List[float]:  # pylint: disable=invalid-name
        """Supported repetition times for device in seconds."""
        # only raise rep_time warning once
        if not PulseBackendConfiguration._rep_time_warning_done:
            warnings.warn('`rep_time` now has units of seconds(s) rather '
                          'than microseconds(mu s).')
            PulseBackendConfiguration._rep_time_warning_done = True

        return self._rep_times

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
        For instance, the backend could be configured with this setting:
            u_channel_lo = [
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]
            ]
        Then, this method can be used as follows:
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
