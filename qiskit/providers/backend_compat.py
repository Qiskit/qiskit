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

from __future__ import annotations
import logging
from typing import List, Iterable, Any, Dict, Optional, Union, Tuple

from qiskit.exceptions import QiskitError

from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties

from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.providers.models.backendproperties import Gate as GateSchema

logger = logging.getLogger(__name__)


def convert_to_target(
    configuration: BackendConfiguration,
    properties: Optional[Union[BackendProperties, Dict]] = None,
    defaults: Optional[Union[PulseDefaults, Dict]] = None,
    custom_name_mapping: Optional[Dict[str, Any]] = None,
    add_delay: bool = True,
    filter_faulty: bool = True,
):
    """Decode transpiler target from backend data set.

    This function directly generates ``Target`` instance without generating
    intermediate legacy objects such as ``BackendProperties`` and ``PulseDefaults``.

    .. note::
        Passing in legacy objects like BackendProperties as properties and PulseDefaults
        as defaults will be deprecated in the future.

    Args:
        configuration: Backend configuration as ``BackendConfiguration``
        properties: Backend property dictionary or ``BackendProperties``
        defaults: Backend pulse defaults dictionary or ``PulseDefaults``
        custom_name_mapping: A name mapping must be supplied for the operation
        not included in Qiskit Standard Gate name mapping, otherwise the operation
        will be dropped in the resulting ``Target`` object.
        add_delay: If True, adds delay to the instruction set.
        filter_faulty: If True, this filters the non-operational qubits.

    Returns:
        A ``Target`` instance.
    """

    # importing pacakges where they are needed, to avoid cyclic-import.
    from qiskit.transpiler.target import Target
    from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, SwitchCaseOp, WhileLoopOp
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
    from qiskit.qobj.pulse_qobj import PulseLibraryItem
    from qiskit.qobj.converters.pulse_instruction import QobjToInstructionConverter
    from qiskit.providers.models.pulsedefaults import Command
    from qiskit.pulse.calibration_entries import PulseQobjDef
    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.gate import Gate

    required = ["measure", "delay"]
    if isinstance(defaults, PulseDefaults):
        defaults = defaults.to_dict()

    if isinstance(properties, BackendProperties):
        properties = properties.to_dict()

    # Load Qiskit object representation
    qiskit_inst_mapping = get_standard_gate_name_mapping()
    if custom_name_mapping:
        qiskit_inst_mapping.update(custom_name_mapping)

    qiskit_control_flow_mapping = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }

    in_data = {"num_qubits": configuration.n_qubits}

    # Parse global configuration properties
    if hasattr(configuration, "dt"):
        in_data["dt"] = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        in_data.update(configuration.timing_constraints)

    # Create instruction property placeholder from backend configuration
    basis_gates = set(getattr(configuration, "basis_gates", []))
    gate_configs = {gate.name: gate for gate in configuration.gates}
    inst_name_map = {}  # type: Dict[str, Instruction]
    prop_name_map = {}  # type: Dict[str, Dict[Tuple[int, ...], InstructionProperties]]
    all_instructions = set.union(basis_gates, set(required))

    faulty_qubits = set()
    faulty_ops = set()
    unsupported_instructions = []

    # Create name to Qiskit instruction object repr mapping
    for name in all_instructions:
        if name in qiskit_control_flow_mapping:
            continue
        if name in qiskit_inst_mapping:
            inst_name_map[name] = qiskit_inst_mapping[name]
        elif name in gate_configs:
            this_config = gate_configs[name]
            params = list(map(Parameter, getattr(this_config, "parameters", [])))
            coupling_map = getattr(this_config, "coupling_map", [])
            inst_name_map[name] = Gate(
                name=name,
                num_qubits=len(coupling_map[0]) if coupling_map else 0,
                params=params,
            )
        else:
            logger.warning(
                "Definition of instruction %s is not found in the Qiskit namespace and "
                "GateConfig is not provided by the BackendConfiguration payload. "
                "Qiskit Gate model cannot be instantiated for this instruction and "
                "this instruction is silently excluded from the Target. "
                "Please add new gate class to Qiskit or provide GateConfig for this name.",
                name,
            )
            unsupported_instructions.append(name)

    for name in unsupported_instructions:
        all_instructions.remove(name)

    # Create empty inst properties from gate configs
    for name, spec in gate_configs.items():
        if hasattr(spec, "coupling_map"):
            coupling_map = spec.coupling_map
            prop_name_map[name] = dict.fromkeys(map(tuple, coupling_map))
        else:
            prop_name_map[name] = None

    # Populate instruction properties
    if properties:
        qubit_properties = list(map(_decode_qubit_property, properties["qubits"]))
        in_data["qubit_properties"] = qubit_properties
        faulty_qubits = {
            q for q, (prop, oper) in enumerate(qubit_properties) if filter_faulty and not oper
        }

        for gate_spec in map(GateSchema.from_dict, properties["gates"]):
            name = gate_spec.gate
            qubits = tuple(gate_spec.qubits)
            if name not in all_instructions:
                logger.info(
                    "Gate property for instruction %s on qubits %s is found "
                    "in the BackendProperties payload. However, this gate is not included in the "
                    "basis_gates or supported_instructions, or maybe the gate model "
                    "is not defined in the Qiskit namespace. This gate is ignored.",
                    name,
                    qubits,
                )
                continue
            inst_prop, operational = _decode_instruction_property(gate_spec)
            if filter_faulty and (set.intersection(faulty_qubits, qubits) or not operational):
                faulty_ops.add((name, qubits))
                try:
                    del prop_name_map[name][qubits]
                except KeyError:
                    pass
                except TypeError:
                    pass
                continue
            if prop_name_map[name] is None:
                prop_name_map[name] = {}
            prop_name_map[name][qubits] = inst_prop
        # Measure instruction property is stored in qubit property in IBM
        measure_props = list(map(_decode_measure_property, properties["qubits"]))
        prop_name_map["measure"] = {}
        for qubit, measure_prop in enumerate(measure_props):
            if qubit in faulty_qubits:
                continue
            qubits = (qubit,)
            prop_name_map["measure"][qubits] = measure_prop

    # Special case for real IBM backend. They don't have delay in gate configuration.
    if add_delay and "delay" not in prop_name_map:
        prop_name_map["delay"] = {
            (q,): None for q in range(configuration.num_qubits) if q not in faulty_qubits
        }

    # Define pulse qobj converter and command sequence for lazy conversion
    if defaults:
        pulse_lib = list(map(PulseLibraryItem.from_dict, defaults["pulse_library"]))
        converter = QobjToInstructionConverter(pulse_lib)
        for cmd in map(Command.from_dict, defaults["cmd_def"]):
            name = cmd.name
            qubits = tuple(cmd.qubits)
            if (
                name not in all_instructions
                or name not in prop_name_map
                or qubits not in prop_name_map[name]
            ):
                logger.info(
                    "Gate calibration for instruction %s on qubits %s is found "
                    "in the PulseDefaults payload. However, this entry is not defined in "
                    "the gate mapping of Target. This calibration is ignored.",
                    name,
                    qubits,
                )
                continue

            if (name, qubits) in faulty_ops:
                continue

            entry = PulseQobjDef(converter=converter, name=cmd.name)
            entry.define(cmd.sequence)
            try:
                prop_name_map[name][qubits].calibration = entry
            except AttributeError:
                logger.info(
                    "The PulseDefaults payload received contains an instruction %s on "
                    "qubits %s which is not present in the configuration or properties payload.",
                    name,
                    qubits,
                )

    # Add parsed properties to target
    target = Target(**in_data)
    for inst_name in all_instructions:
        if inst_name in qiskit_control_flow_mapping:
            # Control flow operator doesn't have gate property.
            target.add_instruction(
                instruction=qiskit_control_flow_mapping[inst_name],
                name=inst_name,
            )
        else:
            target.add_instruction(
                instruction=inst_name_map[inst_name],
                properties=prop_name_map.get(inst_name, None),
            )

    return target


def _decode_qubit_property(qubit_specs: List[Dict]) -> Tuple[QubitProperties, bool]:
    """Decode qubit property data to generate QubitProperty instance.

    Args:
        qubit_specs: List of qubit property dictionary.

    Returns:
        An ``QubitProperty`` instance.
    """
    in_data = {}
    operational = True
    for spec in qubit_specs:
        name = (spec["name"]).lower()
        if name == "operational":
            operational = bool(spec["value"])
        elif name in QubitProperties.__slots__:
            in_data[name] = apply_prefix(value=spec["value"], unit=spec.get("unit", None))
    return QubitProperties(**in_data), operational  # type: ignore[no-untyped-call]


def _decode_instruction_property(
    gate_spec: GateSchema,
):
    """Decode gate property data to generate InstructionProperties instance.

    Args:
        gate_spec: List of gate property dictionary.

    Returns:
        An ``InstructionProperties`` instance and a boolean value representing
        if this gate is operational.
    """

    # importing pacakges where they are needed, to avoid cyclic-import.
    from qiskit.transpiler.target import InstructionProperties

    in_data = {}
    operational = True
    for param in gate_spec.parameters:
        if param.name == "gate_error":
            in_data["error"] = param.value
        if param.name == "gate_length":
            in_data["duration"] = apply_prefix(value=param.value, unit=param.unit)
        if param.name == "operational" and not param.value:
            operational = bool(param.value)
    return InstructionProperties(**in_data), operational


def _decode_measure_property(qubit_specs: List[Dict]):
    """Decode qubit property data to generate InstructionProperties instance.

    Args:
        qubit_specs: List of qubit property dictionary.

    Returns:
        An ``InstructionProperties`` instance.
    """

    # importing pacakges where they are needed, to avoid cyclic-import.
    from qiskit.transpiler.target import InstructionProperties

    in_data = {}
    for spec in qubit_specs:
        name = spec["name"]
        if name == "readout_error":
            in_data["error"] = spec["value"]
        if name == "readout_length":
            in_data["duration"] = apply_prefix(value=spec["value"], unit=spec.get("unit", None))
    return InstructionProperties(**in_data)


def qubit_props_list_from_props(
    properties: BackendProperties,
) -> List[QubitProperties]:
    """Uses BackendProperties to construct
    and return a list of QubitProperties.
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
            frequency = None
        qubit_props.append(
            QubitProperties(  # type: ignore[no-untyped-call]
                t1=t_1,
                t2=t_2,
                frequency=frequency,
            )
        )
    return qubit_props


def convert_to_target_legacy(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
    custom_name_mapping: Optional[Dict[str, Any]] = None,
    add_delay: bool = False,
    filter_faulty: bool = False,
):
    """Uses configuration, properties and pulse defaults
    to construct and return Target class.

    In order to convert with a ``defaults.instruction_schedule_map``,
    which has a custom calibration for an operation,
    the operation name must be in ``configuration.basis_gates`` and
    ``custom_name_mapping`` must be supplied for the operation.
    Otherwise, the operation will be dropped in the resulting ``Target`` object.

    That suggests it is recommended to add custom calibrations **after** creating a target
    with this function instead of adding them to ``defaults`` in advance. For example::

        target.add_instruction(custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)})
    """
    # pylint: disable=cyclic-import
    from qiskit.transpiler.target import (
        Target,
        InstructionProperties,
    )

    from qiskit.circuit.measure import Measure
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

    # Standard gates library mapping, multicontrolled gates not included since they're
    # variable width
    name_mapping = get_standard_gate_name_mapping()
    target = None
    if custom_name_mapping is not None:
        name_mapping.update(custom_name_mapping)
    faulty_qubits = set()
    # Parse from properties if it exsits
    if properties is not None:
        if filter_faulty:
            faulty_qubits = set(properties.faulty_qubits())
        qubit_properties = qubit_props_list_from_props(properties=properties)
        target = Target(
            num_qubits=configuration.n_qubits,
            qubit_properties=qubit_properties,
            concurrent_measurements=getattr(configuration, "meas_map", None),
        )
        # Parse instructions
        gates: Dict[str, Any] = {}
        for gate in properties.gates:
            name = gate.gate
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            else:
                raise QiskitError(
                    f"Operation name {name} does not have a known mapping. Use "
                    "custom_name_mapping to map this name to an Operation object"
                )

            qubits = tuple(gate.qubits)
            if filter_faulty:
                if any(not properties.is_qubit_operational(qubit) for qubit in qubits):
                    continue
                if not properties.is_gate_operational(name, gate.qubits):
                    continue

            gate_props = {}
            for param in gate.parameters:
                if param.name == "gate_error":
                    gate_props["error"] = param.value
                if param.name == "gate_length":
                    gate_props["duration"] = apply_prefix(param.value, param.unit)
            gates[name][qubits] = InstructionProperties(**gate_props)
        for gate, props in gates.items():
            inst = name_mapping[gate]
            target.add_instruction(inst, props)
        # Create measurement instructions:
        measure_props = {}
        for qubit, _ in enumerate(properties.qubits):
            if filter_faulty:
                if not properties.is_qubit_operational(qubit):
                    continue
            try:
                duration = properties.readout_length(qubit)
            except BackendPropertyError:
                duration = None
            try:
                error = properties.readout_error(qubit)
            except BackendPropertyError:
                error = None
            measure_props[(qubit,)] = InstructionProperties(
                duration=duration,
                error=error,
            )
        target.add_instruction(Measure(), measure_props)
    # Parse from configuration because properties doesn't exist
    else:
        target = Target(
            num_qubits=configuration.n_qubits,
            concurrent_measurements=getattr(configuration, "meas_map", None),
        )
        for gate in configuration.gates:
            name = gate.name
            gate_props = (
                {tuple(x): None for x in gate.coupling_map}  # type: ignore[misc]
                if hasattr(gate, "coupling_map")
                else {None: None}
            )
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                raise QiskitError(
                    f"Operation name {name} does not have a known mapping. "
                    "Use custom_name_mapping to map this name to an Operation object"
                )
        target.add_instruction(Measure())
    # parse global configuration properties
    if hasattr(configuration, "dt"):
        target.dt = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        target.granularity = configuration.timing_constraints.get("granularity")
        target.min_length = configuration.timing_constraints.get("min_length")
        target.pulse_alignment = configuration.timing_constraints.get("pulse_alignment")
        target.acquire_alignment = configuration.timing_constraints.get("acquire_alignment")
    # If a pulse defaults exists use that as the source of truth
    if defaults is not None:
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qpbj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in target:
                    if inst == "measure":
                        for qubit in qargs:
                            if filter_faulty and qubit in faulty_qubits:
                                continue
                            target[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in target[inst]:
                        if filter_faulty and any(qubit in faulty_qubits for qubit in qargs):
                            continue
                        target[inst][qargs].calibration = calibration_entry
    combined_global_ops = set()
    if configuration.basis_gates:
        combined_global_ops.update(configuration.basis_gates)
    for op in combined_global_ops:
        if op not in target:
            if op in name_mapping:
                target.add_instruction(name_mapping[op], name=op)
            else:
                raise QiskitError(
                    f"Operation name '{op}' does not have a known mapping. Use "
                    "custom_name_mapping to map this name to an Operation object"
                )
    if add_delay and "delay" not in target:
        target.add_instruction(
            name_mapping["delay"],
            {(bit,): None for bit in range(target.num_qubits) if bit not in faulty_qubits},
        )
    return target


class BackendV2Converter(BackendV2):
    """A converter class that takes a :class:`~.BackendV1` instance and wraps it in a
    :class:`~.BackendV2` interface.

    This class implements the :class:`~.BackendV2` interface and is used to enable
    common access patterns between :class:`~.BackendV1` and :class:`~.BackendV2`. This
    class should only be used if you need a :class:`~.BackendV2` and still need
    compatibility with :class:`~.BackendV1`.

    When using custom calibrations (or other custom workflows) it is **not** recommended
    to mutate the ``BackendV1`` object before applying this converter. For example, in order to
    convert a ``BackendV1`` object with a customized ``defaults().instruction_schedule_map``,
    which has a custom calibration for an operation, the operation name must be in
    ``configuration().basis_gates`` and ``name_mapping`` must be supplied for the operation.
    Otherwise, the operation will be dropped in the resulting ``BackendV2`` object.

    Instead it is typically better to add custom calibrations **after** applying this converter
    instead of updating ``BackendV1.defaults()`` in advance. For example::

        backend_v2 = BackendV2Converter(backend_v1)
        backend_v2.target.add_instruction(
            custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)}
        )
    """

    def __init__(
        self,
        backend: BackendV1,
        name_mapping: Optional[Dict[str, Any]] = None,
        add_delay: bool = True,
        filter_faulty: bool = False,
    ):
        """Initialize a BackendV2 converter instance based on a BackendV1 instance.

        Args:
            backend: The input :class:`~.BackendV1` based backend to wrap in a
                :class:`~.BackendV2` interface
            name_mapping: An optional dictionary that maps custom gate/operation names in
                ``backend`` to an :class:`~.Operation` object representing that
                gate/operation. By default most standard gates names are mapped to the
                standard gate object from :mod:`qiskit.circuit.library` this only needs
                to be specified if the input ``backend`` defines gates in names outside
                that set.
            add_delay: If set to true a :class:`~qiskit.circuit.Delay` operation
                will be added to the target as a supported operation for all
                qubits
            filter_faulty: If the :class:`~.BackendProperties` object (if present) for
                ``backend`` has any qubits or gates flagged as non-operational filter
                those from the output target.
        """
        self._backend = backend
        self._config = self._backend.configuration()
        super().__init__(
            provider=backend.provider,
            name=backend.name(),
            description=self._config.description,
            online_date=getattr(self._config, "online_date", None),
            backend_version=self._config.backend_version,
        )
        self._options = self._backend._options
        self._properties = None
        if hasattr(self._backend, "properties"):
            self._properties = self._backend.properties()
        self._defaults = None
        self._target = None
        self._name_mapping = name_mapping
        self._add_delay = add_delay
        self._filter_faulty = filter_faulty

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
            self._target = convert_to_target(
                configuration=self._config,
                properties=self._properties,
                defaults=self._defaults,
                custom_name_mapping=self._name_mapping,
                add_delay=self._add_delay,
                filter_faulty=self._filter_faulty,
            )
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
        return self._config.meas_map

    def drive_channel(self, qubit: int):
        return self._config.drive(qubit)

    def measure_channel(self, qubit: int):
        return self._config.measure(qubit)

    def acquire_channel(self, qubit: int):
        return self._config.acquire(qubit)

    def control_channel(self, qubits: Iterable[int]):
        return self._config.control(qubits)

    def run(self, run_input, **options):
        return self._backend.run(run_input, **options)
