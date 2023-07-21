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

"""
Utilities for constructing Target object from configuration, properties and
pulse defaults json files
"""

from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models.pulsedefaults import PulseDefaults


def convert_to_target(conf_dict: dict, props_dict: dict = None, defs_dict: dict = None) -> Target:
    """Uses configuration, properties and pulse defaults dicts
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
    qubit_props = None
    if props_dict:
        qubit_props = qubit_props_from_props(props_dict)
    target = Target(qubit_properties=qubit_props, concurrent_measurements=conf_dict.get("meas_map"))
    # Parse from properties if it exsits
    if props_dict is not None:
        # Parse instructions
        gates = {}
        for gate in props_dict["gates"]:
            name = gate["gate"]
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            elif name not in custom_gates:
                custom_gate = Gate(name, len(gate["qubits"]), [])
                custom_gates[name] = custom_gate
                gates[name] = {}

            qubits = tuple(gate["qubits"])
            gate_props = {}
            for param in gate["parameters"]:
                if param["name"] == "gate_error":
                    gate_props["error"] = param["value"]
                if param["name"] == "gate_length":
                    gate_props["duration"] = apply_prefix(param["value"], param["unit"])
            gates[name][qubits] = InstructionProperties(**gate_props)
        for gate, props in gates.items():
            if gate in name_mapping:
                inst = name_mapping.get(gate)
            else:
                inst = custom_gates[gate]
            target.add_instruction(inst, props)
        # Create measurement instructions:
        measure_props = {}
        count = 0
        for qubit in props_dict["qubits"]:
            qubit_prop = {}
            for prop in qubit:
                if prop["name"] == "readout_length":
                    qubit_prop["duration"] = apply_prefix(prop["value"], prop["unit"])
                if prop["name"] == "readout_error":
                    qubit_prop["error"] = prop["value"]
            measure_props[(count,)] = InstructionProperties(**qubit_prop)
            count += 1
        target.add_instruction(Measure(), measure_props)
    # Parse from configuration because properties doesn't exist
    else:
        for gate in conf_dict["gates"]:
            name = gate["name"]
            gate_props = {tuple(x): None for x in gate["coupling_map"]}
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                custom_gate = Gate(name, len(gate["coupling_map"][0]), [])
                target.add_instruction(custom_gate, gate_props)
        measure_props = {(n,): None for n in range(conf_dict["n_qubits"])}
        target.add_instruction(Measure(), measure_props)
    # parse global configuration properties
    dt = conf_dict.get("dt")
    if dt:
        target.dt = dt * 1e-9
    if "timing_constraints" in conf_dict:
        target.granularity = conf_dict["timing_constraints"].get("granularity")
        target.min_length = conf_dict["timing_constraints"].get("min_length")
        target.pulse_alignment = conf_dict["timing_constraints"].get("pulse_alignment")
        target.acquire_alignment = conf_dict["timing_constraints"].get("acquire_alignment")
    # If pulse defaults exists use that as the source of truth
    if defs_dict is not None:
        # TODO remove the usage of PulseDefaults as it will be deprecated in the future
        pulse_defs = PulseDefaults.from_dict(defs_dict)
        inst_map = pulse_defs.instruction_schedule_map
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
                            target[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in target[inst]:
                        target[inst][qargs].calibration = calibration_entry
    target.add_instruction(
        Delay(Parameter("t")), {(bit,): None for bit in range(target.num_qubits)}
    )
    return target


def qubit_props_from_props(properties: dict) -> list:
    """Returns a dictionary of `qiskit.providers.backend.QubitProperties` using
    a backend properties dictionary created by loading props.json payload.
    """
    qubit_props = []
    for qubit in properties["qubits"]:
        qubit_properties = {}
        for prop_dict in qubit:
            if prop_dict["name"] == "T1":
                qubit_properties["t1"] = apply_prefix(prop_dict["value"], prop_dict["unit"])
            elif prop_dict["name"] == "T2":
                qubit_properties["t2"] = apply_prefix(prop_dict["value"], prop_dict["unit"])
            elif prop_dict["name"] == "frequency":
                qubit_properties["frequency"] = apply_prefix(prop_dict["value"], prop_dict["unit"])
        qubit_props.append(QubitProperties(**qubit_properties))
    return qubit_props
