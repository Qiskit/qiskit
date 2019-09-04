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
# TODO:
#  - get_property
#  - units for gate info
#  - describe channel
#  - parameterized schedule
#  - SWITCH OP TO CMD?
#  - Separate out CmdDef?
#  - draw
#  - __getattr__
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from qiskit.qobj.converters import QobjToInstructionConverter

from qiskit.pulse.channels import *
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule

QubitArgument = Union[int, Iterable[int]]


class SystemInfo(object):
    """A resource for getting information from a backend, tailored for Pulse users."""

    def __init__(self,
                 backend: 'BaseBackend',
                 default_ops: Optional[Dict[Tuple[str, int, ...], Schedule]]):
        """
        Initialize a SystemInfo instance with the data from the backend.

        Args:
            backend: A Pulse enabled backend returned by a Qiskit provider.
            default_ops: {(op_name, *qubits): `Schedule` or `ParameterizedSchedule`}
        """
        if not backend.configuration().open_pulse:
            raise PulseError("The backend '{}' is not enabled "
                             "with OpenPulse.".format(backend.name()))
        self._backend = backend
        self._defaults = backend.defaults()
        self._properties = backend.properties()
        self._config = backend.configuration()
        for key, schedule in default_ops.items():
            self.add_op(key[0], key[1:], schedule)

    @property
    def name(self):
        """The name given to this system."""
        return self._properties.backend_name

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
        try:
            return self._coupling_map
        except AttributeError:
            self._coupling_map = defaultdict(set)
            for control, target in self._config.coupling_map:
                self._coupling_map[control].add(target)
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

    def get_property(self, name: str, *args): #-> Tuple[datetime.time, float]:
        """Return the collected time and value of the property, if it was given by the backend."""
        # TODO: this kinda works but not great
        ret = self._properties.__dict__[name]
        for arg in args:
            ret = ret[arg]
        return ret

    def gate_error(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return gate error estimates from backend properties.

        Args:
            operation: The operation for which to get the error.
            qubits: The specific qubits for the operation.
        Raises:
            PulseError: If the gate error is not provided.
        """
        qubits = _to_tuple(qubits)
        try:
            gate_info = self._gates[operation][qubits]
        except KeyError:
            raise PulseError("Gate {} on qubits {} is not defined.".format(operation, qubits))
        try:
            return gate_info['params']['gate_error']
        except KeyError:
            raise PulseError("There is no error estimate provided for the {} gate on this "
                             "system.".format(gate_info['name']))

    def gate_length(self, operation: str, qubits: Union[int, Iterable[int]]) -> float:
        """
        Return the duration of the gate in units of dt.

        Args:
            operation: The operation for which to get the duration.
            qubits: The specific qubits for the operation.
        Raises:
            PulseError: If the gate length is not provided.
        """
        qubits = _to_tuple(qubits)
        try:
            gate_info = self._gates[operation][qubits]
        except KeyError:
            raise PulseError("Gate {} on qubits {} is not defined.".format(operation, qubits))
        try:
            return gate_info['params']['gate_error']
        except KeyError:
            raise PulseError("There is no information provided about the duration of the {} "
                             "gate on this system.".format(gate_info['name']))

    def drives(self, qubit: int) -> DriveChannel:
        """Return the drive channel for the given qubit."""
        return DriveChannel(qubit)

    def measures(self, qubit: int) -> MeasureChannel:
        """Return the measure stimulus channel for the given qubit."""
        return MeasureChannel(qubit)

    def acquires(self, qubit: int) -> AcquireChannel:
        """Return the acquisition channel for the given qubit."""
        return AcquireChannel(qubit)

    def controls(self, qubit: int) -> ControlChannel:
        """Return the control channel for the given qubit."""
        # TODO
        return ControlChannel(qubit)

    def describe(self, channel: Channel):
        # TODO
        pass

    @property
    def ops(self) -> List[str]:
        """
        Return all operations which are defined by default. (This is essentially the basis gates
        along with measure and reset.)
        """
        return [k for k in self._ops_definition.keys()]

    def op_qubits(self, operation: str) -> List[Union[int, Tuple[int]]]:
        """
        Return a list of the qubits for which the given operation is defined. Single qubit
        operations return a flat list, and multiqubit operations return a list of tuples.
        """
        return [qs[0] if len(qs) == 1 else qs for qs in self._ops_definition[operation].keys()]

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
        # TODO fixme
        sched = self._ops_definition[operation].get(_to_tuple(qubits))
        if sched is None:
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))
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
        if not qubits:
            raise PulseError("Cannot add definition {} with no target qubits.".format(operation))
        self._ops_definition[operation][_to_tuple(qubits)] = schedule

    def remove_op(self, operation: str, qubits: Union[int, List[int]]):
        """Remove the given operation from the defined operations."""
        self._ops_definition[operation].pop(_to_tuple(qubits))

    def draw(self) -> None:
        """
        Visualize the topology of the device, showing qubits, their interconnections, and the
        channels which interact with them. Optionally print a listing of the supported 1Q and
        2Q gates.
        """
        # TODO
        raise NotImplementedError

    @property
    def _qubit_ops(self) -> Dict[Tuple[int], List[str]]:
        """
        A lazily evaluated mapping from qubits to the operations defined on them:
            {(qubits): [operations]}
        """
        try:
            return self.__qubit_ops
        except AttributeError:
            self._process_cmd_def()
            return self.__qubit_ops

    @property
    def _ops_definition(self) -> Dict[str, Dict[Tuple[int], Schedule]]:
        """
        A lazily evaluated singly nested mapping from operations, to qubits, to Schedules:
            {operation:
                {(qubits): Schedule}}
        """
        try:
            return self.__ops_definition
        except AttributeError:
            self._process_cmd_def()
            return self.__ops_definition

    def _process_cmd_def(self) -> None:
        """
        Reformat the command definition from the backend defaults to fill the _ops_definition
        and the backwards lookup table _qubit_ops.
        """
        self.__ops_definition = defaultdict(dict)
        self.__qubit_ops = defaultdict(list)
        converter = QobjToInstructionConverter(self._defaults.pulse_library,
                                               buffer=self.buffer)
        for op in self._defaults.cmd_def:
            self.add_op(
                op.name,
                op.qubits,
                ParameterizedSchedule(*[converter(inst) for inst in op.sequence], name=op.name))
            self.__qubit_ops[_to_tuple(op.qubits)].append(op.name)

    @property
    def _gates(self) -> Dict[str, Dict[Tuple[int], Dict[str, Any]]]:
        """Lazy reformatting of gate properties."""
        try:
            return self.__gates
        except AttributeError:
            self.__gates = defaultdict(dict)
            for gate in self._properties.gates:
                qubits = _to_tuple(qubits)
                # TODO: units!
                params = {param.name: param.value for param in gate.parameters}
                self.__gates[gate.gate][qubits] = {
                    'name': gate.name,
                    'params': params
                }
            return self.__gates

    def __str__(self) -> str:
        return '{}({} qubit{} {})'.format(self.name,
                                          self.n_qubits,
                                          's' if self.n_qubits > 1 else '',
                                          self.basis_gates)

    # def __getattr__(self, attr) -> Any:
    #     """
    #     Capture undefined attribute lookups and interpret it as an operation
    #     lookup. Priority goes to `get_property' as defined earlier, but
    #     collisions are not expected.
    #         For example:
    #             system.x(0) <=> system.get(`x', qubit=0)
    #     def draw(self) -> None:
    #     Capture undefined attribute lookups and interpret them as backend
    #     properties.
    #         For example:
    #             system.backend_name <=> system.get_property(backend_name)
    #             system.t1(0) <=> system.get_property('t1', 0)
    #     """
    #     def fancy_get(qubits: Union[int, Iterable[int]] = None,
    #                   *params: List[Union[int, float, complex]],
    #                   **kwparams: Dict[str, Union[int, float, complex]]):
    #         try:
    #             return self.get(attr, qubits=qubits, *params, **kwparams)
    #         except PulseError:
    #             # return self.get_property(attr, args)
    #             raise AttributeError("{} object has no attribute "
    #                                  "'{}'".format(self.__class__.__name__, attr))
    #     return fancy_get

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
