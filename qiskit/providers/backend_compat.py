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

"""Backend abstract interface for providers."""


from typing import List, Union, Iterable, Any, Dict

from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.exceptions import BackendPropertyError


def convert_to_target(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
) -> Target:
    """Uses configuration, properties and pulse defaults
    to construct and return Target class.
    """
    name_mapping = {
        "id": IGate(),
        "sx": SXGate(),
        "x": XGate(),
        "cx": CXGate(),
        "rz": RZGate(Parameter("λ")),
        "reset": Reset(),
    }
    custom_gates = {}
    target = None
    # Parse from properties if it exsits
    if properties is not None:
        qubit_properties = qubit_props_list_from_props(properties=properties)
        target = Target(num_qubits=configuration.n_qubits, qubit_properties=qubit_properties)
        # Parse instructions
        gates: Dict[str, Any] = {}
        for gate in properties.gates:
            name = gate.gate
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            elif name not in custom_gates:
                custom_gate = Gate(name, len(gate.qubits), [])
                custom_gates[name] = custom_gate
                gates[name] = {}

            qubits = tuple(gate.qubits)
            gate_props = {}
            for param in gate.parameters:
                if param.name == "gate_error":
                    gate_props["error"] = param.value
                if param.name == "gate_length":
                    gate_props["duration"] = apply_prefix(param.value, param.unit)
            gates[name][qubits] = InstructionProperties(**gate_props)
        for gate, props in gates.items():
            if gate in name_mapping:
                inst = name_mapping.get(gate)
            else:
                inst = custom_gates[gate]
            target.add_instruction(inst, props)
        # Create measurement instructions:
        measure_props = {}
        for qubit, _ in enumerate(properties.qubits):
            measure_props[(qubit,)] = InstructionProperties(
                duration=properties.readout_length(qubit),
                error=properties.readout_error(qubit),
            )
        target.add_instruction(Measure(), measure_props)
    # Parse from configuration because properties doesn't exist
    else:
        target = Target(num_qubits=configuration.n_qubits)
        for gate in configuration.gates:
            name = gate.name
            gate_props = (
                {tuple(x): None for x in gate.coupling_map}  # type: ignore[misc]
                if hasattr(gate, "coupling_map")
                else {None: None}
            )
            gate_len = len(gate.coupling_map[0]) if hasattr(gate, "coupling_map") else 0
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                custom_gate = Gate(name, gate_len, [])
                target.add_instruction(custom_gate, gate_props)
        target.add_instruction(Measure())
    # parse global configuration properties
    if hasattr(configuration, "dt"):
        target.dt = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        target.granularity = configuration.timing_constraints.get("granularity")
        target.min_length = configuration.timing_constraints.get("min_length")
        target.pulse_alignment = configuration.timing_constraints.get("pulse_alignment")
        target.aquire_alignment = configuration.timing_constraints.get("acquire_alignment")
    # If a pulse defaults exists use that as the source of truth
    if defaults is not None:
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                sched = inst_map.get(inst, qarg)
                if inst in target:
                    try:
                        qarg = tuple(qarg)
                    except TypeError:
                        qarg = (qarg,)
                    if inst == "measure":
                        for qubit in qarg:
                            target[inst][(qubit,)].calibration = sched
                    elif qarg in target[inst]:
                        target[inst][qarg].calibration = sched
    if "delay" not in target:
        target.add_instruction(
            Delay(Parameter("t")), {(bit,): None for bit in range(target.num_qubits)}
        )
    return target


def qubit_props_list_from_props(
    properties: BackendProperties,
) -> List[QubitProperties]:
    """Uses BackendProperties to construct
    and return a list of IBMQubitProperties.
    """
    qubit_props: List[QubitProperties] = []
    for qubit, _ in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except BackendPropertyError:
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except BackendPropertyError:
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except BackendPropertyError:
            t_2 = None
        qubit_props.append(
            QubitProperties(  # type: ignore[no-untyped-call]
                t1=t_1,
                t2=t_2,
                frequency=frequency,
            )
        )
    return qubit_props


class BackendV2Converter(BackendV2):
    """A converter class that takes a :class:`~.BackendV1` instance and wraps it in a
    :class:`~.BackendV2` interface.

    This class implements the :class:`~.BackendV2` interface and is used to enable
    common access patterns between :class:`~.BackendV1` and :class:`~.BackendV2`. This
    class should only be used if you need a :class:`~.BackendV2` and still need
    compatibility with :class:`~.BackendV1`.
    """

    def __init__(
        self,
        backend: BackendV1,
    ):
        """Initialize a BackendV2 converter instance based on a BackendV1 instance.

        Args:
            backend: The input :class:`~.BackendV1` based backend to wrap in a
                :class:`~.BackendV2` interface
        """
        self._backend = backend
        self._config = self._backend.configuration()
        super().__init__(
            provider=backend.provider,
            name=backend.name(),
            description=self._config.description,
            online_date=self._config.online_date,
            backend_version=self._config.backend_version,
        )
        self._options = self._backend._options
        self._properties = None
        if hasattr(self._backend, "properties"):
            self._properties = self._backend.properties()
        self._defaults = None
        self._target = None

    @property
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        if self._target is None:
            if self._defaults is None and hasattr(self._backend, "defaults"):
                self._defaults = self._backend.defaults()
            if self._properties is None and hasattr(self._backend, "properties"):
                self._properties = self._backend.properties()
            self._target = convert_to_target(self._config, self._properties, self._defaults)
        return self._target

    @property
    def max_circuits(self):
        return self._config.max_experiments

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def dtm(self) -> float:
        return self._config.dtm

    @property
    def meas_map(self) -> List[List[int]]:
        return self._config.dt

    def drive_channel(self, qubit: int):
        self._config.drive(qubit)

    def measure_channel(self, qubit: int):
        self._config.measure(qubit)

    def acquire_channel(self, qubit: int):
        self._config.acquire(qubit)

    def control_channel(self, qubits: Iterable[int]):
        self._config.control(qubits)

    def run(self, run_input, **options):
        return self._backend.run(run_input, **options)
