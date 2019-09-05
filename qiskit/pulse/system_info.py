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

"""
This object provides an efficient, centralized interface for extracting backend information useful
to building Pulse schedules, streamlining this part of our schedule building workflow. It is
important to note that the resulting `Schedule` and its execution are not constrainted by the
SystemInfo. For constraint validation, see the `validate.py` module (coming soon).

Questions about the backend which can be answered by SystemInfo and are likely to come up when
building schedules include:
  - What is the topology of this backend?
  - What characteristics (e.g. T1, T2) do the qubits on this backend have?
  - What is the time delta between signal samples (`dt`) on this backend?
  - What channel should be used to drive qubit 0?
  - What are the defined native gates on this backend?
"""
import datetime
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from qiskit.qobj.converters import QobjToInstructionConverter

from qiskit.pulse.channels import (Channel, DriveChannel, MeasureChannel, ControlChannel,
                                   AcquireChannel)
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule

# pylint: disable=missing-return-doc
# Questions
#   - CmdDef should be separate so that backend can be required?
#   - Memoize? Or make QobjToInstructionConverter faster?
#   - __getattr__ and get_property are not stable, and should also have more tests


class SystemInfo():
    """A resource for getting information from a backend, tailored for Pulse users."""

    def __init__(self,
                 backend: Optional['BaseBackend'] = None,
                 default_ops: Optional[Dict[Tuple[str, int], Schedule]] = None):
        """
        Initialize a SystemInfo instance with the data from the backend.

        Args:
            backend: A Pulse enabled backend returned by a Qiskit provider.
            default_ops: {(op_name, *qubits): `Schedule` or `ParameterizedSchedule`}
        """
        # FIXME
        # This stores the circuit operation definitions
        self._ops_definition = defaultdict(dict)
        # This is a helpful backwards mapping from qubits -> defined operations
        self._qubit_ops = defaultdict(list)
        # Both of the above definitions will be filled in by defaults if backend is provided

        self._backend = backend
        if backend:
            if not backend.configuration().open_pulse:
                raise PulseError("The backend '{}' is not enabled "
                                 "with OpenPulse.".format(backend.name()))
            self._backend_props = backend.properties()
            self._defaults = backend.defaults()
            self._config = backend.configuration()
            self._process_backend_props()
            self._process_defaults()
            self._process_config()
        if default_ops:
            for key, schedule in default_ops.items():
                self.add_op(key[0], key[1:], schedule)

    @property
    def name(self):
        """The name given to this system."""
        return self._backend_props.backend_name

    @property
    def version(self):
        """The name given to this system."""
        return self._backend_props.backend_version

    @property
    def n_qubits(self) -> int:
        """The number of total qubits on the device."""
        return self._config.n_qubits

    @property
    def dt(self) -> float:
        """Time delta between samples on the signal channels in seconds."""
        return self._config.dt * 1.e-9

    @property
    def dtm(self) -> float:
        """Time delta between samples on the acquisition channels in seconds."""
        return self._config.dtm * 1e-9

    @property
    def sample_rate(self) -> float:
        """Sample rate of the signal channels in Hz (1/dt)."""
        return 1.0 / self.dt

    @property
    def basis_gates(self) -> List[str]:
        """Return the gate operations which are defined by default."""
        return self._config.basis_gates

    @property
    def coupling_map(self) -> Dict[int, Set[int]]:
        """The adjacency list of available multiqubit operations."""
        return self._coupling_map

    @property
    def meas_map(self) -> List[List[int]]:
        """
        Return the measurement groups. The measurement groups are lists of qubits that must be
        measured together. For example, if the device has 3 qubits, and the measurement map is
        [[0], [1, 2]], then qubit 0 can be measured alone, but qubit 1 is always measured with
        qubit 2 and visa versa.

        To get a default measurement schedule from this, you could use, for instance:
        acquire_sched = sysinfo.get('measure', sysinfo.meas_map[1])
        """
        return self._config.meas_map

    @property
    def buffer(self) -> int:
        """Default delay time between pulses in units of dt."""
        return self._defaults.buffer

    def hamiltonian(self) -> str:
        """
        Return the LaTeX Hamiltonian string for this device and print its description if
        provided.

        Raises:
            PulseError: If the hamiltonian is not defined.
        """
        ham = self._config.hamiltonian.get('h_latex')
        if ham is None:
            raise PulseError("Hamiltonian not found.")
        print(self._config.hamiltonian.get('description'))
        return ham

    def qubit_freq_est(self, qubit: int) -> float:
        """
        Return the estimated resonant frequency for the given qubit in Hz.

        Args:
            qubit: Index of the qubit of interest.
        Raises:
            PulseError: If the frequency is not available.
        """
        try:
            return self._defaults.qubit_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the qubit frequency for qubit {qub}, this system only "
                             "has {num} qubits.".format(qub=qubit, num=self.n_qubits))

    def meas_freq_est(self, qubit: int) -> float:
        """
        Return the estimated measurement stimulus frequency to readout from the given qubit.

        Args:
            qubit: Index of the qubit of interest.
        Raises:
            PulseError: If the frequency is not available.
        """
        try:
            return self._defaults.meas_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the measurement frequency for qubit {qub}, this system "
                             "only has {num} qubits.".format(qub=qubit, num=self.n_qubits))

    def drives(self, qubit: int) -> DriveChannel:
        """
        Return the drive channel for the given qubit.

        Raises:
            PulseError: If the qubit is not a part of the system.
        """
        if qubit > self.n_qubits:
            raise PulseError("This system does not have {} qubits.".format(qubit))
        return DriveChannel(qubit)

    def measures(self, qubit: int) -> MeasureChannel:
        """
        Return the measure stimulus channel for the given qubit.

        Raises:
            PulseError: If the qubit is not a part of the system.
        """
        if qubit > self.n_qubits:
            raise PulseError("This system does not have {} qubits.".format(qubit))
        return MeasureChannel(qubit)

    def acquires(self, qubit: int) -> AcquireChannel:
        """
        Return the acquisition channel for the given qubit.

        Raises:
            PulseError: If the qubit is not a part of the system.
        """
        if qubit > self.n_qubits:
            raise PulseError("This system does not have {} qubits.".format(qubit))
        return AcquireChannel(qubit)

    def controls(self, qubit: int) -> ControlChannel:
        """
        Return the control channel for the given qubit.

        Raises:
            PulseError: If the qubit is not a part of the system.
        """
        # TODO: It's probable that controls can't map trivially to qubits.
        if qubit > self.n_qubits:
            raise PulseError("This system does not have {} qubits.".format(qubit))
        return ControlChannel(qubit)

    def describe(self, channel: ControlChannel) -> Dict[Channel, complex]:
        """
        Return a basic description of the channel dependency. Derived channels are given weights
        which describe how their frames are linked to other frames.

        For instance, the backend could be configured with this setting:
            u_channel_lo = [
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]
            ]
        Then, given that sysinfo is SystemInfo(backend):
            sysinfo.describe(ControlChannel(1))
            >>> {DriveChannel(0): -1, DriveChannel(1): 1}

        Args:
            channel: The derived channel to describe.
        Raises:
            PulseError: If channel is not a ControlChannel.
        """
        if not isinstance(channel, ControlChannel):
            raise PulseError("Can only describe ControlChannels.")
        result = {}
        for u_chan_lo in self._config.u_channel_lo[channel.index]:
            result[DriveChannel(u_chan_lo.q)] = u_chan_lo.scale
        return result

    def get_property(self,
                     name: str,
                     *args: List[Union[int, str]],
                     error: bool = False) -> Union[None, Tuple[Any, datetime.datetime]]:
        """
        Return the value and collected time of the property if it was given by the backend,
        otherwise, return `None` or raise an error.

        Args:
            name: The property to look for.
            args: Optionally used to specify within the heirarchy which property to return.
            error: If True, then raise an error when the property is not found.
        Raises:
            PulseError: If error is True and the property is not found.
        """
        # FIXME
        try:
            ret = self._props.get(name)
            for arg in args:
                try:
                    if isinstance(ret, str) or isinstance(ret, list):
                        # wont fail if overspecified
                        return ret
                    ret = ret[arg]
                except KeyError:
                    # This can help if one of args is an integer or list of qubits
                    ret = ret[_to_tuple(arg)]
        except (KeyError, TypeError):
            if error:
                raise PulseError("Could not find the desired property.")
            else:
                return None
        return ret

    def gate_error(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return gate error estimates from backend properties.

        Args:
            operation: The operation for which to get the error.
            qubits: The specific qubits for the operation.
        """
        # Throw away datetime at index 1
        return self.get_property('gates',
                                 operation,
                                 _to_tuple(qubits),
                                 'gate_error',
                                 error=True)[0]

    def gate_length(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return the duration of the gate in units of seconds.

        Args:
            operation: The operation for which to get the duration.
            qubits: The specific qubits for the operation.
        """
        # Throw away datetime at index 1
        return self.get_property('gates',
                                 operation,
                                 _to_tuple(qubits),
                                 'gate_length',
                                 error=True)[0]

    @property
    def ops(self) -> List[str]:
        """
        Return all operations which are defined by default. (This is essentially the basis gates
        along with measure and reset.)
        """
        return list(self._ops_definition.keys())

    def op_qubits(self, operation: str) -> List[Union[int, Tuple[int]]]:
        """
        Return a list of the qubits for which the given operation is defined. Single qubit
        operations return a flat list, and multiqubit operations return a list of tuples.
        """
        return [qs[0] if len(qs) == 1 else qs
                for qs in sorted(self._ops_definition[operation].keys())]

    def qubit_ops(self, qubits: Union[int, List[int]]) -> List[str]:
        """
        Return a list of the operation names that are defined by the backend for the given qubit
        or qubits.
        """
        return self._qubit_ops[_to_tuple(qubits)]

    def has(self, operation: str, qubits: Union[int, Iterable[int]]) -> bool:
        """
        Is the operation defined for the given qubits?

        Args:
            operation: The operation for which to look.
            qubits: The specific qubits for the operation.
        """
        return operation in self._ops_definition and \
            _to_tuple(qubits) in self._ops_definition[operation]

    def get(self,
            operation: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """
        Return the defined Schedule for the given operation on the given qubits.

        Args:
            operation: Name of the operation.
            qubits: The qubits for the operation.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Raises:
            PulseError: If the operation is not defined on the qubits.
        """
        qubits = _to_tuple(qubits)
        if not self.has(operation, qubits):
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))
        sched = self._ops_definition[operation].get(qubits)
        if isinstance(sched, ParameterizedSchedule):
            sched = sched.bind_parameters(*params, **kwparams)
        return sched

    def get_parameters(self, operation: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """
        Return the list of parameters taken by the given operation on the given qubits.

        Raises:
            PulseError: If the operation is not defined on the qubits.
        """
        qubits = _to_tuple(qubits)
        if not self.has(operation, qubits):
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))
        return self._ops_definition[operation][qubits].parameters

    def add_op(self, operation: str, qubits: Union[int, Iterable[int]], schedule: Schedule):
        """
        Add a new known operation.

        Args:
            operation: The name of the operation to add.
            qubits: The qubits which the operation applies to.
            schedule: The Schedule that implements the given operation.
        Raises:
            PulseError: If the qubits are provided as an empty iterable.
        """
        qubits = _to_tuple(qubits)
        if qubits == ():
            raise PulseError("Cannot add definition {} with no target qubits.".format(operation))
        if not (isinstance(schedule, Schedule) or isinstance(schedule, ParameterizedSchedule)):
            raise PulseError("Attemping to add an invalid schedule type.")
        self._ops_definition[operation][qubits] = schedule

    def remove_op(self, operation: str, qubits: Union[int, List[int]]):
        """Remove the given operation from the defined operations."""
        qubits = _to_tuple(qubits)
        if not self.has(operation, qubits):
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))
        self._ops_definition[operation].pop(qubits)

    def draw(self) -> None:
        """
        Visualize the topology of the device, showing qubits, their interconnections, and the
        channels which interact with them. Optionally print a listing of the supported 1Q and
        2Q gates.
        """
        # TODO: Implement the draw method.
        raise NotImplementedError

    def _process_defaults(self) -> None:
        """
        Reformat the command definition from the backend defaults to fill the _ops_definition
        and the backwards lookup table _qubit_ops.
        """
        converter = QobjToInstructionConverter(self._defaults.pulse_library,
                                               buffer=self.buffer)
        for op in self._defaults.cmd_def:
            qubits = _to_tuple(op.qubits)
            self.add_op(
                op.name,
                qubits,
                ParameterizedSchedule(*[converter(inst) for inst in op.sequence], name=op.name))
            self._qubit_ops[qubits].append(op.name)

    def _process_config(self) -> None:
        """
        Reformat the backend provided configuration.

        Defines:
            _coupling_map: An adjacency list exposed to the user through `coupling_map`.
        """
        self._coupling_map = defaultdict(set)
        for control, target in self._config.coupling_map:
            self._coupling_map[control].add(target)

    def _process_backend_props(self) -> None:
        """
        Fill in a reformatted version of backend properties as `_props`, extracting values for
        gate and qubit properties. For example:
            {
                'backend_name': 'ibmq_device',
                'backend_version': '0.0.0',
                'gates': {
                    'id': {(0,): {
                        'gate_error': (0.001, datetime.datetime(...)),
                        'gate_length': (1.1e-07, datetime.datetime(...))
                    }
                    ...
                }
                'qubits': {
                    0: {
                        'T1': (5.5e-05, datetime.datetime(...)),
                        'readout_error': (0.029, datetime.datetime(...))
                        ...
                    }
                    ...
                }
                ...
            }
        """
        def apply_prefix(value, unit):
            prefixes = {
               'p': 1e-12, 'n': 1e-9,
               'u': 1e-6,  'µ': 1e-6,
               'm': 1e-3,  'k': 1e3,
               'M': 1e6,   'G': 1e9,
            }
            if not unit:
                return value
            try:
                return value * prefixes[unit[0]]
            except KeyError:
                raise PulseError("Could not understand units: {}".format(unit))

        props = {}
        props.update(self._backend_props.__dict__)

        props['gates'] = defaultdict(dict)
        for gate in self._backend_props.gates:
            qubits = _to_tuple(gate.qubits)
            gate_props = {}
            for param in gate.parameters:
                value = apply_prefix(param.value, param.unit)
                gate_props[param.name] = (value, param.date)
            gate_props['name'] = gate.name
            props['gates'][gate.gate][qubits] = gate_props

        props['qubits'] = defaultdict(dict)
        for qubit, params in enumerate(self._backend_props.qubits):
            qubit_props = {}
            for param in params:
                value = apply_prefix(param.value, param.unit)
                qubit_props[param.name] = (value, param.date)
            props['qubits'][qubit] = qubit_props
        self._props = props

    def __str__(self) -> str:
        if not self._backend:
            return object.__str__(self)
        return '{}({} qubit{} operating on {})'.format(
            self.name,
            self.n_qubits,
            's' if self.n_qubits > 1 else '',
            self.basis_gates)

    def __repr__(self) -> str:
        if not self._backend:
            return object.__repr__(self)
        ops = {op: qubs.keys() for op, qubs in self._ops_definition.items()}
        ham = self._config.hamiltonian.get('description') if self._config.hamiltonian else ''
        return ("{}({} {}Q\n    Operations:\n{}\n    Properties:\n{}\n    Configuration:\n{}\n"
                "    Hamiltonian:\n{})".format(self.__class__.__name__,
                                               self.name,
                                               self.n_qubits,
                                               ops,
                                               list(self._props.keys()),
                                               list(self._config.__dict__.keys()),
                                               ham))

    def __getattr__(self, attr: str) -> Any:
        """
        Capture undefined attribute lookups and interpret it as an operation
        lookup.
            For example:
                system.x(0) <=> system.get(`x', qubit=0)
        Capture undefined attribute lookups and interpret them as backend
        properties.
            For example:
                system.backend_name <=> system.get_property(backend_name)
                system.t1(0) <=> system.get_property('t1', 0)
        """
        # FIXME
        if self._backend is None:
            raise PulseError("Please instantiate the SystemInfo with a backend to get this "
                             "information.")

        def fancy_get(qubits: Union[int, Iterable[int]] = None,
                      *params: List[Union[int, float, complex]],
                      **kwparams: Dict[str, Union[int, float, complex]]):
            try:
                qubits = _to_tuple(qubits)
                return self.get(attr, qubits, *params, **kwparams)
            except PulseError:
                try:
                    return self.get_property(attr, *params, error=True)
                except PulseError:
                    raise AttributeError("{} object has no attribute "
                                         "'{}'".format(self.__class__.__name__, attr))
        return fancy_get


def _to_tuple(values: Union[int, Iterable[int]]) -> Tuple[int]:
    """
    Return the input, sorted, and as a tuple.

    Args:
        values: An integer, a list of ints, or a tuple of ints.
    """
    try:
        return tuple(sorted(values))
    except TypeError:
        return (values,)
