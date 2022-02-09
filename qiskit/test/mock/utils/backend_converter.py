import pprint
import sys
import json
import warnings

from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.test.mock.utils.json_decoder import decode_pulse_defaults


def convert_to_target(configuration, properties=None, defaults=None):
    name_mapping = {
        "id": IGate(),
        "sx": SXGate(),
        "x": XGate(),
        "cx": CXGate(),
        "rz": RZGate(Parameter("λ")),
        "reset": Reset(),
    }
    custom_gates = {}
    target = Target()
    # Parse from properties if it exsits
    if properties is not None:
        # Parse instructions
        gates = {}
        for gate in properties["gates"]:
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
                inst = name_mapping[gate]
            else:
                inst = custom_gates[gate]
            target.add_instruction(inst, props)
        # Create measurement instructions:
        measure_props = {}
        count = 0
        for qubit in properties["qubits"]:
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
        for gate in configuration["gates"]:
            name = gate["name"]
            gate_props = {tuple(x): None for x in gate["coupling_map"]}
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                custom_gate = Gate(name, len(gate["coupling_map"][0]))
                target.add_instruction(custom_gate, gate_props)
        measure_props = {(n,): None for n in range(configuration["n_qubits"])}
        target.add_instruction(Measure(), measure_props)
    # parse global configuration properties
    dt = configuration.get("dt")
    if dt:
        target.dt = dt ** 1e-6
    if "timing_constraints" in configuration:
        target.granularity = configuration["timing_constraints"].get("granularity")
        target.min_length = configuration["timing_constraints"].get("min_length")
        target.pulse_alignment = configuration["timing_constraints"].get("pulse_alignment")
        target.aquire_alignment = configuration["timing_constraints"].get("aquire_alignment")
    # If a pulse defaults exists use that as the source of truth
    # TODO: uncomment when measurement qargs fix is applied
    #    if defaults is not None:
    #        decode_pulse_defaults(defaults)
    #        pulse_defs = PulseDefaults.from_dict(defaults)
    #        target.update_from_instruction_schedule_map(pulse_defs.instruction_schedule_map)
    return target


def qubit_properties_dict_from_properties(properties: dict):
    count = 0
    qubit_props = {}
    for qubit in properties["qubits"]:
        qubit_properties = {}
        for prop_dict in qubit:
            if prop_dict["name"] == "T1":
                qubit_properties["t1"] = apply_prefix(prop_dict["value"], prop_dict["unit"])
            elif prop_dict["name"] == "T2":
                qubit_properties["t2"] = apply_prefix(prop_dict["value"], prop_dict["unit"])
            elif prop_dict["name"] == "frequency":
                qubit_properties["frequency"] = apply_prefix(
                    prop_dict["value"], prop_dict["unit"]
                )
        qubit_props[count] = QubitProperties(**qubit_properties)
        count += 1
    return qubit_props


def main():
    conf_path = sys.argv[1]
    properties_path = None
    if len(sys.argv) > 2:
        properties_path = sys.argv[2]
    defaults_path = None
    if len(sys.argv) > 3:
        defaults_path = sys.argv[3]
    with open(conf_path, "r") as fd:
        conf_dict = json.load(fd)
    defaults_dict = None
    if defaults_path:
        with open(defaults_path, "r") as fd:
            defaults_dict = json.load(fd)
    properties_dict = None
    if properties_path:
        with open(properties_path, "r") as fd:
            properties_dict = json.load(fd)
    target = convert_to_target(conf_dict, properties_dict, defaults_dict)
    if properties_dict is not None:
        qubit_properties = qubit_properties_dict_from_properties(properties_dict)
    else:
        qubit_properties = {}
    print(target)
    pprint.pprint(qubit_properties)


if __name__ == "__main__":
    main()
