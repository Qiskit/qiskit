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
from typing import List, Iterable, Any, Dict, Optional, Tuple

from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties

from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError

logger = logging.getLogger(__name__)


def convert_to_target(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
    custom_name_mapping: Optional[Dict[str, Any]] = None,
    add_delay: bool = True,
    filter_faulty: bool = True,
):
    """Decode transpiler target from backend data set.

    This function generates ``Target`` instance from intermediate
    legacy objects such as ``BackendProperties`` and ``PulseDefaults``.

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
    # pylint: disable=cyclic-import
    from qiskit.transpiler.target import (
        Target,
        InstructionProperties,
    )
    from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, SwitchCaseOp, WhileLoopOp
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.gate import Gate

    required = ["measure", "delay"]

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

    faulty_ops = set()
    faulty_qubits = []
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
        qubit_properties = [
            QubitProperties(
                t1=properties.qubit_property(qubit_idx)["T1"][0],
                t2=properties.qubit_property(qubit_idx)["T2"][0],
                frequency=properties.qubit_property(qubit_idx)["frequency"][0],
            )
            for qubit_idx in range(0, configuration.num_qubits)
        ]

        in_data["qubit_properties"] = qubit_properties

        if filter_faulty:
            faulty_qubits = properties.faulty_qubits()

        for name in prop_name_map.keys():
            for qubits, params in properties.gate_property(name).items():
                in_param = {
                    "error": params["gate_error"][0] if "gate_error" in params else None,
                    "duration": params["gate_length"][0] if "gate_length" in params else None,
                }
                inst_prop = InstructionProperties(**in_param)

                if filter_faulty and (
                    (not properties.is_gate_operational(name, qubits))
                    or any(not properties.is_qubit_operational(qubit) for qubit in qubits)
                ):
                    faulty_ops.add((name, qubits))
                    try:
                        del prop_name_map[name][qubits]
                    except KeyError:
                        pass
                    continue

                if prop_name_map[name] is None:
                    prop_name_map[name] = {}

                prop_name_map[name][qubits] = inst_prop

        # Measure instruction property is stored in qubit property
        prop_name_map["measure"] = {}

        for qubit_idx in range(configuration.num_qubits):
            if qubit_idx in faulty_qubits:
                continue
            qubit_prop = properties.qubit_property(qubit_idx)
            in_prop = {
                "duration": qubit_prop["readout_length"][0]
                if "readout_length" in qubit_prop
                else None,
                "error": qubit_prop["readout_error"][0] if "readout_error" in qubit_prop else None,
            }
            prop_name_map["measure"][(qubit_idx,)] = InstructionProperties(**in_prop)

    if add_delay and "delay" not in prop_name_map:
        prop_name_map["delay"] = {
            (q,): None for q in range(configuration.num_qubits) if q not in faulty_qubits
        }

    if defaults:
        inst_sched_map = defaults.instruction_schedule_map

        for name in inst_sched_map.instructions:
            for qubits in inst_sched_map.qubits_with_instruction(name):

                if not isinstance(qubits, tuple):
                    qubits = (qubits,)

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

                entry = inst_sched_map._get_calibration_entry(name, qubits)

                try:
                    prop_name_map[name][qubits].calibration = entry
                except AttributeError:
                    logger.info(
                        "The PulseDefaults payload received contains an instruction %s on "
                        "qubits %s which is not present in the configuration or properties payload.",
                        name,
                        qubits,
                    )

    # Remove 'delay' if add_delay is set to False.
    if not add_delay:
        if "delay" in all_instructions:
            all_instructions.remove("delay")

    # Add parsed properties to target
    target = Target(**in_data)
    for inst_name in all_instructions:
        if inst_name in qiskit_control_flow_mapping:
            # Control flow operator doesn't have gate property.
            target.add_instruction(
                instruction=qiskit_control_flow_mapping[inst_name],
                name=inst_name,
            )
        elif properties is None:
            target.add_instruction(
                instruction=inst_name_map[inst_name],
                name=inst_name,
            )
        else:
            target.add_instruction(
                instruction=inst_name_map[inst_name],
                properties=prop_name_map.get(inst_name, None),
            )

    return target


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
        filter_faulty: bool = True,
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
        self._defaults = None

        if hasattr(self._backend, "properties"):
            self._properties = self._backend.properties()
        if hasattr(self._backend, "defaults"):
            self._defaults = self._backend.defaults()

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
