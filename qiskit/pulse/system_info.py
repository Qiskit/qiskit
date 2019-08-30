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
  - What is the time delta between samples (`dt`) on this backend?
  - What channel should I use to drive qubit 0?
  - What are the defined native gates on this backend?
"""
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union

from qiskit.qobj.converters import QobjToInstructionConverter

from qiskit.pulse.channels import *
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule


class SystemInfo(object):
    """A resource for getting information from a backend, tailored for Pulse users."""

    def __init__(self, backend: 'BaseBackend'):
        """
        Initialize a SystemInfo instance with the data from the backend.

        Args:
            backend: A Pulse enabled backend returned by a Qiskit provider.
        """
        if not backend.configuration().open_pulse:
            raise PulseError("The backend '{}' is not enabled "
                             "with OpenPulse.".format(backend.name()))

        self._backend = backend
        self._defaults = backend.defaults()
        self._properties = backend.properties()
        self._config = backend.configuration()
        self._ops_definition = defaultdict(dict)
        converter = QobjToInstructionConverter(self._defaults.pulse_library,
                                               buffer=self._defaults.buffer)
        for cmd in self._defaults.cmd_def:
            self.add_op(
                cmd.name,
                cmd.qubits,
                ParameterizedSchedule(*[converter(inst) for inst in cmd.sequence], name=cmd.name))

    @property
    def n_qubits(self) -> int:
        """The number of total qubits on the device."""
        return self._config.n_qubits
    
    @property
    def coupling_map(self) -> Dict[int, List[int]]:
        """The adjacency list of available multiqubit operations."""
        if self._coupling_map:
            return self._coupling_map
        self._coupling_map = defaultdict(set)
        for control, target in self._config.coupling_map:
            self._coupling_map[control].add(target)
        return self._coupling_map

    @property
    def hamiltonian(self) -> str:
        """Return the LaTeX Hamiltonian string for this device."""
        print(self._config.hamiltonian.get('description'))
        return self._config.hamiltonian.get('h_latex', PulseError("Hamiltonian not found."))
    
    @property
    def buffer(self) -> int:
        """Default delay time between pulses in units of dt."""
        return self._defaults.buffer
    
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

    def qubit_freq_est(self, qubit: int) -> float:
        """Return the estimated resonant frequency for the given qubit in Hz."""
        try:
            return self._defaults.qubit_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the qubit frequency for qubit {qub}, this system only "
                             "has {num} qubits.".format(qub=qubit, num=self.n_qubits))
    
    def meas_freq_est(self, qubit: int) -> float:
        """Return the estimated measurement stimulus frequency to readout from the given qubit."""
        try:
            return self._defaults.meas_freq_est[qubit] * 1e9
        except IndexError:
            raise PulseError("Cannot get the qubit frequency for qubit {qub}, this system only "
                             "has {num} qubits.".format(qub=qubit, num=self.n_qubits))

    def get_property(self, name: str, *args): #-> Tuple[datetime.time, float]:
        """Return the collected time and value of the property, if it was given by the backend."""
        pass

    def gate_error(self, operation: str, qubits: List[int]) -> float:
        """Return gate error estimates from backend properties."""
        pass
    
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
        pass
    
    def describe(self, channel: Channel):
        pass

    @property
    def basis_gates(self) -> List[str]:
        """Return the gate operations which are defined by default."""
        return self._config.basis_gates
    
    @property
    def ops(self) -> List[str]:
        """
        Return all operations which are defined by default. (This is essentially the basis gates
        along with measure and reset.)
        """
        return [k for k in self._ops_definition.keys()]

    def op_qubits(self, operation: str) -> List[List[int]]:
        """Return a list of lists of the qubits for which the given op is defined."""
        return [k for k in self._ops_definition[operation].keys()]

    def has(self, operation: str, qubits: Union[int, List[int]]) -> bool:
        """Is the operation defined for the given qubits?"""
        return operation in self._ops_definition and qubits in self._ops_definition[operation]

    def get(self,
            operation: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """Return the defined Schedule for the given op on the given qubits."""
        qubits = tuple(qubits) if isinstance(qubits, list) else qubits
        try:
            sched = self._ops_definition[operation][qubits]
        except KeyError:
            raise PulseError("Operation {op} for qubits {qubits} is not defined for this "
                             "system.".format(op=operation, qubits=qubits))
        # TODO fixme
        if isinstance(sched, ParameterizedSchedule):
            sched = sched.bind_parameters(*params, **kwparams)
        return sched

    def get_parameters(self, operation: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """Return the list of parameters taken by the given operation."""
        pass
    
    def add_op(self, operation: str, qubits: Union[int, List[int]], schedule: Schedule):
        """Add a new known operation."""
        # TODO test tuple(qubits)
        if not qubits:
            raise PulseError("Cannot add definition {} with no target qubits.".format(operation))
        if isinstance(qubits, int) or len(qubits) == 1:
            qubit = qubit if isinstance(qubits, int) else qubits[0]
            self._ops_definition[operation][qubit] = schedule
        else:
            self._ops_definition[operation][tuple(qubits)] = schedule
    
    def remove_op(self, operation: str, qubits: Union[int, List[int]]):
        """Remove the given operation from the defined operations."""
        qubits = tuple(qubits) if isinstance(qubits, list) else qubits
        self._ops_definition[operation].pop(qubits)
    
    def draw(self) -> None:
        """
        "Print a listing of the default device 1Q and 2Q gates."
        Visualize the topology of the device, showing qubits, their interconnections, and the
        channels which interact with them.
        """
        pass

    def __getattr__(self, attr) -> Any:
        """
        Capture undefined attribute lookups and interpret it as an operation
        lookup. Priority goes to `get_property' as defined earlier, but
        collisions are not expected.
            For example:
                system.x(0) <=> system.get(`x', qubit=0)
        def draw(self) -> None:
        Capture undefined attribute lookups and interpret them as backend
        properties.
            For example:
                system.backend_name <=> system.get_property(backend_name)
                system.t1(0) <=> system.get_property('t1', 0)
        """
        def fancy_get(qubits: Union[int, Iterable[int]] = None,
                      *params: List[Union[int, float, complex]],
                      **kwparams: Dict[str, Union[int, float, complex]]):
            try:
                return self.get(attr, qubits=qubits, *params, **kwparams)
            except PulseError:
                # return self.get_property(attr, args)
                raise AttributeError("{} object has no attribute "
                                     "'{}'".format(self.__class__.__name__, attr))
        return fancy_get
