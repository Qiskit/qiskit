# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Converters for migration from IBM Quantum BackendV1 to BackendV2."""

from typing import Any, Dict

from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models import (
    BackendConfiguration,
    BackendProperties,
    PulseDefaults,
)


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
    target = Target(num_qubits=configuration.n_qubits)
    # Parse from properties if it exsits
    if properties is not None:
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
        target.aquire_alignment = configuration.timing_constraints.get(
            "acquire_alignment"
        )
    # If pulse defaults exists use that as the source of truth
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
                    else:
                        target[inst][qarg].calibration = sched
    return target


def qubit_props_dict_from_props(properties: BackendProperties) -> QubitProperties:
    """Uses BackendProperties to construct
    and return QubitProperties class.
    """
    qubit_props = {}
    for qubit, _ in enumerate(properties.qubits):
        qubit_props[qubit] = QubitProperties(
            t1=properties.t1(qubit),
            t2=properties.t2(qubit),
            frequency=properties.frequency(qubit),
        )
    return qubit_props