# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Binary IO for circuit objects."""

import io
import json
import struct
import uuid
import warnings

import numpy as np

from qiskit import circuit as circuit_mod
from qiskit import extensions
from qiskit.circuit import library, controlflow
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.extensions import quantum_initializer
from qiskit.qpy import common, formats, exceptions
from qiskit.qpy.binary_io import value
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.synthesis import evolution as evo_synth


def _read_header_v2(file_obj, version, vectors, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER_V2._make(
        struct.unpack(
            formats.CIRCUIT_HEADER_V2_PACK,
            file_obj.read(formats.CIRCUIT_HEADER_V2_SIZE),
        )
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    global_phase = value.loads_value(
        data.global_phase_type,
        file_obj.read(data.global_phase_size),
        version=version,
        vectors=vectors,
    )
    header = {
        "global_phase": global_phase,
        "num_qubits": data.num_qubits,
        "num_clbits": data.num_clbits,
        "num_registers": data.num_registers,
        "num_instructions": data.num_instructions,
    }
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return header, name, metadata


def _read_header(file_obj, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER._make(
        struct.unpack(formats.CIRCUIT_HEADER_PACK, file_obj.read(formats.CIRCUIT_HEADER_SIZE))
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    header = {
        "global_phase": data.global_phase,
        "num_qubits": data.num_qubits,
        "num_clbits": data.num_clbits,
        "num_registers": data.num_registers,
        "num_instructions": data.num_instructions,
    }
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return header, name, metadata


def _read_registers_v4(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        data = formats.REGISTER_V4._make(
            struct.unpack(
                formats.REGISTER_V4_PACK,
                file_obj.read(formats.REGISTER_V4_SIZE),
            )
        )
        name = file_obj.read(data.name_size).decode("utf8")
        REGISTER_ARRAY_PACK = "!%sq" % data.size
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if data.type.decode("utf8") == "q":
            registers["q"][name] = (data.standalone, bit_indices, data.in_circuit)
        else:
            registers["c"][name] = (data.standalone, bit_indices, data.in_circuit)
    return registers


def _read_registers(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        data = formats.REGISTER._make(
            struct.unpack(
                formats.REGISTER_PACK,
                file_obj.read(formats.REGISTER_SIZE),
            )
        )
        name = file_obj.read(data.name_size).decode("utf8")
        REGISTER_ARRAY_PACK = "!%sI" % data.size
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if data.type.decode("utf8") == "q":
            registers["q"][name] = (data.standalone, bit_indices, True)
        else:
            registers["c"][name] = (data.standalone, bit_indices, True)
    return registers


def _read_instruction_parameter(file_obj, version, vectors):
    type_key, bin_data = common.read_instruction_param(file_obj)

    if type_key == common.ProgramTypeKey.CIRCUIT:
        param = common.data_from_binary(bin_data, read_circuit, version=version)
    elif type_key == common.ContainerTypeKey.RANGE:
        data = formats.RANGE._make(struct.unpack(formats.RANGE_PACK, bin_data))
        param = range(data.start, data.stop, data.step)
    elif type_key == common.ContainerTypeKey.TUPLE:
        param = tuple(
            common.sequence_from_binary(
                bin_data,
                _read_instruction_parameter,
                version=version,
                vectors=vectors,
            )
        )
    elif type_key == common.ValueTypeKey.INTEGER:
        # TODO This uses little endian. Should be fixed in the next QPY version.
        param = struct.unpack("<q", bin_data)[0]
    elif type_key == common.ValueTypeKey.FLOAT:
        # TODO This uses little endian. Should be fixed in the next QPY version.
        param = struct.unpack("<d", bin_data)[0]
    else:
        param = value.loads_value(type_key, bin_data, version, vectors)

    return param


def _read_instruction(file_obj, circuit, registers, custom_instructions, version, vectors):
    instruction = formats.CIRCUIT_INSTRUCTION._make(
        struct.unpack(
            formats.CIRCUIT_INSTRUCTION_PACK,
            file_obj.read(formats.CIRCUIT_INSTRUCTION_SIZE),
        )
    )
    gate_name = file_obj.read(instruction.name_size).decode(common.ENCODE)
    label = file_obj.read(instruction.label_size).decode(common.ENCODE)
    condition_register = file_obj.read(instruction.condition_register_size).decode(common.ENCODE)
    qargs = []
    cargs = []
    params = []
    condition_tuple = None
    if instruction.has_condition:
        # If an invalid register name is used assume it's a single bit
        # condition and treat the register name as a string of the clbit index
        if ClassicalRegister.name_format.match(condition_register) is None:
            # If invalid register prefixed with null character it's a clbit
            # index for single bit condition
            if condition_register[0] == "\x00":
                conditional_bit = int(condition_register[1:])
                condition_tuple = (circuit.clbits[conditional_bit], instruction.condition_value)
            else:
                raise ValueError(
                    f"Invalid register name: {condition_register} for condition register of "
                    f"instruction: {gate_name}"
                )
        else:
            condition_tuple = (registers["c"][condition_register], instruction.condition_value)
    qubit_indices = dict(enumerate(circuit.qubits))
    clbit_indices = dict(enumerate(circuit.clbits))

    # Load Arguments
    for _qarg in range(instruction.num_qargs):
        qarg = formats.CIRCUIT_INSTRUCTION_ARG._make(
            struct.unpack(
                formats.CIRCUIT_INSTRUCTION_ARG_PACK,
                file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE),
            )
        )
        if qarg.type.decode(common.ENCODE) == "c":
            raise TypeError("Invalid input carg prior to all qargs")
        qargs.append(qubit_indices[qarg.size])
    for _carg in range(instruction.num_cargs):
        carg = formats.CIRCUIT_INSTRUCTION_ARG._make(
            struct.unpack(
                formats.CIRCUIT_INSTRUCTION_ARG_PACK,
                file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE),
            )
        )
        if carg.type.decode(common.ENCODE) == "q":
            raise TypeError("Invalid input qarg after all qargs")
        cargs.append(clbit_indices[carg.size])

    # Load Parameters
    for _param in range(instruction.num_parameters):
        param = _read_instruction_parameter(file_obj, version, vectors)
        params.append(param)

    # Load Gate object
    if gate_name in ("Gate", "Instruction"):
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if instruction.label_size > 0:
            inst_obj.label = label
        circuit._append(inst_obj, qargs, cargs)
        return
    elif gate_name in custom_instructions:
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if instruction.label_size > 0:
            inst_obj.label = label
        circuit._append(inst_obj, qargs, cargs)
        return
    elif hasattr(library, gate_name):
        gate_class = getattr(library, gate_name)
    elif hasattr(circuit_mod, gate_name):
        gate_class = getattr(circuit_mod, gate_name)
    elif hasattr(extensions, gate_name):
        gate_class = getattr(extensions, gate_name)
    elif hasattr(quantum_initializer, gate_name):
        gate_class = getattr(quantum_initializer, gate_name)
    elif hasattr(controlflow, gate_name):
        gate_class = getattr(controlflow, gate_name)
    else:
        raise AttributeError("Invalid instruction type: %s" % gate_name)

    if gate_name in {"IfElseOp", "WhileLoopOp"}:
        gate = gate_class(condition_tuple, *params)
    else:
        if gate_name in {"Initialize", "UCRXGate", "UCRYGate", "UCRZGate"}:
            gate = gate_class(params)
        else:
            if gate_name == "Barrier":
                params = [len(qargs)]
            elif gate_name in {"BreakLoopOp", "ContinueLoopOp"}:
                params = [len(qargs), len(cargs)]
            gate = gate_class(*params)
        gate.condition = condition_tuple
    if instruction.label_size > 0:
        gate.label = label
    if not isinstance(gate, Instruction):
        circuit.append(gate, qargs, cargs)
    else:
        circuit._append(gate, qargs, cargs)


def _parse_custom_instruction(custom_instructions, gate_name, params):
    type_str, num_qubits, num_clbits, definition = custom_instructions[gate_name]
    type_key = common.CircuitInstructionTypeKey(type_str)

    if type_key == common.CircuitInstructionTypeKey.INSTRUCTION:
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition is not None:
            inst_obj.definition = definition
        return inst_obj

    if type_key == common.CircuitInstructionTypeKey.GATE:
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
        return inst_obj

    if type_key == common.CircuitInstructionTypeKey.PAULI_EVOL_GATE:
        return definition

    raise ValueError("Invalid custom instruction type '%s'" % type_str)


def _read_pauli_evolution_gate(file_obj, version, vectors):
    pauli_evolution_def = formats.PAULI_EVOLUTION_DEF._make(
        struct.unpack(
            formats.PAULI_EVOLUTION_DEF_PACK, file_obj.read(formats.PAULI_EVOLUTION_DEF_SIZE)
        )
    )
    if pauli_evolution_def.operator_size != 1 and pauli_evolution_def.standalone_op:
        raise ValueError(
            "Can't have a standalone operator with {pauli_evolution_raw[0]} operators in the payload"
        )

    operator_list = []
    for _ in range(pauli_evolution_def.operator_size):
        op_elem = formats.SPARSE_PAULI_OP_LIST_ELEM._make(
            struct.unpack(
                formats.SPARSE_PAULI_OP_LIST_ELEM_PACK,
                file_obj.read(formats.SPARSE_PAULI_OP_LIST_ELEM_SIZE),
            )
        )
        op_raw_data = common.data_from_binary(file_obj.read(op_elem.size), np.load)
        operator_list.append(SparsePauliOp.from_list(op_raw_data))

    if pauli_evolution_def.standalone_op:
        pauli_op = operator_list[0]
    else:
        pauli_op = operator_list

    time = value.loads_value(
        common.ValueTypeKey(pauli_evolution_def.time_type),
        file_obj.read(pauli_evolution_def.time_size),
        version=version,
        vectors=vectors,
    )
    synth_data = json.loads(file_obj.read(pauli_evolution_def.synth_method_size))
    synthesis = getattr(evo_synth, synth_data["class"])(**synth_data["settings"])
    return_gate = library.PauliEvolutionGate(pauli_op, time=time, synthesis=synthesis)
    return return_gate


def _read_custom_instructions(file_obj, version, vectors):
    custom_instructions = {}
    custom_definition_header = formats.CUSTOM_CIRCUIT_DEF_HEADER._make(
        struct.unpack(
            formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK,
            file_obj.read(formats.CUSTOM_CIRCUIT_DEF_HEADER_SIZE),
        )
    )
    if custom_definition_header.size > 0:
        for _ in range(custom_definition_header.size):
            data = formats.CUSTOM_CIRCUIT_INST_DEF._make(
                struct.unpack(
                    formats.CUSTOM_CIRCUIT_INST_DEF_PACK,
                    file_obj.read(formats.CUSTOM_CIRCUIT_INST_DEF_SIZE),
                )
            )
            name = file_obj.read(data.gate_name_size).decode(common.ENCODE)
            type_str = data.type
            definition_circuit = None
            if data.custom_definition:
                def_binary = file_obj.read(data.size)
                if version < 3 or not name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = common.data_from_binary(
                        def_binary, read_circuit, version=version
                    )
                elif name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = common.data_from_binary(
                        def_binary, _read_pauli_evolution_gate, version=version, vectors=vectors
                    )
            custom_instructions[name] = (
                type_str,
                data.num_qubits,
                data.num_clbits,
                definition_circuit,
            )
    return custom_instructions


def _write_instruction_parameter(file_obj, param):
    if isinstance(param, QuantumCircuit):
        type_key = common.ProgramTypeKey.CIRCUIT
        data = common.data_to_binary(param, write_circuit)
    elif isinstance(param, range):
        type_key = common.ContainerTypeKey.RANGE
        data = struct.pack(formats.RANGE_PACK, param.start, param.stop, param.step)
    elif isinstance(param, tuple):
        type_key = common.ContainerTypeKey.TUPLE
        data = common.sequence_to_binary(param, _write_instruction_parameter)
    elif isinstance(param, int):
        # TODO This uses little endian. This should be fixed in next QPY version.
        type_key = common.ValueTypeKey.INTEGER
        data = struct.pack("<q", param)
    elif isinstance(param, float):
        # TODO This uses little endian. This should be fixed in next QPY version.
        type_key = common.ValueTypeKey.FLOAT
        data = struct.pack("<d", param)
    else:
        type_key, data = value.dumps_value(param)
    common.write_instruction_param(file_obj, type_key, data)


# pylint: disable=too-many-boolean-expressions
def _write_instruction(file_obj, instruction_tuple, custom_instructions, index_map):
    gate_class_name = instruction_tuple[0].__class__.__name__
    if (
        (
            not hasattr(library, gate_class_name)
            and not hasattr(circuit_mod, gate_class_name)
            and not hasattr(extensions, gate_class_name)
            and not hasattr(quantum_initializer, gate_class_name)
            and not hasattr(controlflow, gate_class_name)
        )
        or gate_class_name == "Gate"
        or gate_class_name == "Instruction"
        or isinstance(instruction_tuple[0], library.BlueprintCircuit)
    ):
        if instruction_tuple[0].name not in custom_instructions:
            custom_instructions[instruction_tuple[0].name] = instruction_tuple[0]
        gate_class_name = instruction_tuple[0].name

    elif isinstance(instruction_tuple[0], library.PauliEvolutionGate):
        gate_class_name = r"###PauliEvolutionGate_" + str(uuid.uuid4())
        custom_instructions[gate_class_name] = instruction_tuple[0]

    has_condition = False
    condition_register = b""
    condition_value = 0
    if instruction_tuple[0].condition:
        has_condition = True
        if isinstance(instruction_tuple[0].condition[0], Clbit):
            bit_index = index_map["c"][instruction_tuple[0].condition[0]]
            condition_register = b"\x00" + str(bit_index).encode(common.ENCODE)
            condition_value = int(instruction_tuple[0].condition[1])
        else:
            condition_register = instruction_tuple[0].condition[0].name.encode(common.ENCODE)
            condition_value = instruction_tuple[0].condition[1]

    gate_class_name = gate_class_name.encode(common.ENCODE)
    label = getattr(instruction_tuple[0], "label")
    if label:
        label_raw = label.encode(common.ENCODE)
    else:
        label_raw = b""
    instruction_raw = struct.pack(
        formats.CIRCUIT_INSTRUCTION_PACK,
        len(gate_class_name),
        len(label_raw),
        len(instruction_tuple[0].params),
        instruction_tuple[0].num_qubits,
        instruction_tuple[0].num_clbits,
        has_condition,
        len(condition_register),
        condition_value,
    )
    file_obj.write(instruction_raw)
    file_obj.write(gate_class_name)
    file_obj.write(label_raw)
    file_obj.write(condition_register)
    # Encode instruciton args
    for qbit in instruction_tuple[1]:
        instruction_arg_raw = struct.pack(
            formats.CIRCUIT_INSTRUCTION_ARG_PACK, b"q", index_map["q"][qbit]
        )
        file_obj.write(instruction_arg_raw)
    for clbit in instruction_tuple[2]:
        instruction_arg_raw = struct.pack(
            formats.CIRCUIT_INSTRUCTION_ARG_PACK, b"c", index_map["c"][clbit]
        )
        file_obj.write(instruction_arg_raw)
    # Encode instruction params
    for param in instruction_tuple[0].params:
        _write_instruction_parameter(file_obj, param)


def _write_pauli_evolution_gate(file_obj, evolution_gate):
    operator_list = evolution_gate.operator
    standalone = False
    if not isinstance(operator_list, list):
        operator_list = [operator_list]
        standalone = True
    num_operators = len(operator_list)

    def _write_elem(buffer, op):
        elem_data = common.data_to_binary(op.to_list(array=True), np.save)
        elem_metadata = struct.pack(formats.SPARSE_PAULI_OP_LIST_ELEM_PACK, len(elem_data))
        buffer.write(elem_metadata)
        buffer.write(elem_data)

    pauli_data_buf = io.BytesIO()
    for operator in operator_list:
        data = common.data_to_binary(operator, _write_elem)
        pauli_data_buf.write(data)

    time_type, time_data = value.dumps_value(evolution_gate.time)
    time_size = len(time_data)
    synth_class = str(type(evolution_gate.synthesis).__name__)
    settings_dict = evolution_gate.synthesis.settings
    synth_data = json.dumps({"class": synth_class, "settings": settings_dict}).encode(common.ENCODE)
    synth_size = len(synth_data)
    pauli_evolution_raw = struct.pack(
        formats.PAULI_EVOLUTION_DEF_PACK,
        num_operators,
        standalone,
        time_type,
        time_size,
        synth_size,
    )
    file_obj.write(pauli_evolution_raw)
    file_obj.write(pauli_data_buf.getvalue())
    pauli_data_buf.close()
    file_obj.write(time_data)
    file_obj.write(synth_data)


def _write_custom_instruction(file_obj, name, instruction):
    type_key = common.CircuitInstructionTypeKey.assign(instruction)
    has_definition = False
    size = 0
    data = None
    num_qubits = instruction.num_qubits
    num_clbits = instruction.num_clbits

    if type_key == common.CircuitInstructionTypeKey.PAULI_EVOL_GATE:
        has_definition = True
        data = common.data_to_binary(instruction, _write_pauli_evolution_gate)
        size = len(data)
    elif instruction.definition is not None:
        has_definition = True
        data = common.data_to_binary(instruction.definition, write_circuit)
        size = len(data)
    name_raw = name.encode(common.ENCODE)
    custom_instruction_raw = struct.pack(
        formats.CUSTOM_CIRCUIT_INST_DEF_PACK,
        len(name_raw),
        type_key,
        num_qubits,
        num_clbits,
        has_definition,
        size,
    )
    file_obj.write(custom_instruction_raw)
    file_obj.write(name_raw)
    if data:
        file_obj.write(data)


def _write_registers(file_obj, in_circ_regs, full_bits):
    bitmap = {bit: index for index, bit in enumerate(full_bits)}
    processed_indices = set()

    out_circ_regs = set()
    for bit in full_bits:
        if bit._register is not None and bit._register not in in_circ_regs:
            out_circ_regs.add(bit._register)

    for regs, is_in_circuit in [(in_circ_regs, True), (out_circ_regs, False)]:
        for reg in regs:
            standalone = all(bit._register is reg for bit in reg)
            reg_name = reg.name.encode(common.ENCODE)
            reg_type = reg.prefix.encode(common.ENCODE)
            file_obj.write(
                struct.pack(
                    formats.REGISTER_V4_PACK,
                    reg_type,
                    standalone,
                    reg.size,
                    len(reg_name),
                    is_in_circuit,
                )
            )
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = "!%sq" % reg.size
            bit_indices = []
            for bit in reg:
                bit_index = bitmap.get(bit, -1)
                if bit_index in processed_indices:
                    bit_index = -1
                if bit_index >= 0:
                    processed_indices.add(bit_index)
                bit_indices.append(bit_index)
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *bit_indices))

    return len(in_circ_regs) + len(out_circ_regs)


def write_circuit(file_obj, circuit, metadata_serializer=None):
    """Write a single QuantumCircuit object in the file like object.

    Args:
        file_obj (FILE): The file like object to write the circuit data in.
        circuit (QuantumCircuit): The circuit data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.QuantumCircuit.metadata` dictionary for
            each circuit in ``circuits`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
    """
    metadata_raw = json.dumps(
        circuit.metadata, separators=(",", ":"), cls=metadata_serializer
    ).encode(common.ENCODE)
    metadata_size = len(metadata_raw)
    num_instructions = len(circuit)
    circuit_name = circuit.name.encode(common.ENCODE)
    global_phase_type, global_phase_data = value.dumps_value(circuit.global_phase)

    with io.BytesIO() as reg_buf:
        num_qregs = _write_registers(reg_buf, circuit.qregs, circuit.qubits)
        num_cregs = _write_registers(reg_buf, circuit.cregs, circuit.clbits)
        registers_raw = reg_buf.getvalue()
    num_registers = num_qregs + num_cregs

    # Write circuit header
    header_raw = formats.CIRCUIT_HEADER_V2(
        name_size=len(circuit_name),
        global_phase_type=global_phase_type,
        global_phase_size=len(global_phase_data),
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        metadata_size=metadata_size,
        num_registers=num_registers,
        num_instructions=num_instructions,
    )
    header = struct.pack(formats.CIRCUIT_HEADER_V2_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(global_phase_data)
    file_obj.write(metadata_raw)
    # Write header payload
    file_obj.write(registers_raw)
    instruction_buffer = io.BytesIO()
    custom_instructions = {}
    index_map = {}
    index_map["q"] = {bit: index for index, bit in enumerate(circuit.qubits)}
    index_map["c"] = {bit: index for index, bit in enumerate(circuit.clbits)}
    for instruction in circuit.data:
        _write_instruction(instruction_buffer, instruction, custom_instructions, index_map)
    file_obj.write(struct.pack(formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK, len(custom_instructions)))

    for name, instruction in custom_instructions.items():
        _write_custom_instruction(file_obj, name, instruction)

    instruction_buffer.seek(0)
    file_obj.write(instruction_buffer.read())
    instruction_buffer.close()


def read_circuit(file_obj, version, metadata_deserializer=None):
    """Read a single QuantumCircuit object from the file like object.

    Args:
        file_obj (FILE): The file like object to read the circuit data from.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.QuantumCircuit.metadata` attribute for any circuits
            in the QPY file. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.

    Returns:
        QuantumCircuit: The circuit object from the file.

    Raises:
        QpyError: Invalid register.
    """
    vectors = {}
    if version < 2:
        header, name, metadata = _read_header(file_obj, metadata_deserializer=metadata_deserializer)
    else:
        header, name, metadata = _read_header_v2(
            file_obj, version, vectors, metadata_deserializer=metadata_deserializer
        )

    global_phase = header["global_phase"]
    num_qubits = header["num_qubits"]
    num_clbits = header["num_clbits"]
    num_registers = header["num_registers"]
    num_instructions = header["num_instructions"]
    out_registers = {"q": {}, "c": {}}
    if num_registers > 0:
        circ = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        if version < 4:
            registers = _read_registers(file_obj, num_registers)
        else:
            registers = _read_registers_v4(file_obj, num_registers)

        for bit_type_label, bit_type, reg_type in [
            ("q", Qubit, QuantumRegister),
            ("c", Clbit, ClassicalRegister),
        ]:
            register_bits = set()
            # Add quantum registers and bits
            for register_name in registers[bit_type_label]:
                standalone, indices, in_circuit = registers[bit_type_label][register_name]
                indices_defined = [x for x in indices if x >= 0]
                # If a register has no bits in the circuit skip it
                if not indices_defined:
                    continue
                if standalone:
                    start = min(indices_defined)
                    count = start
                    out_of_order = False
                    for index in indices:
                        if index < 0:
                            out_of_order = True
                            continue
                        if not out_of_order and index != count:
                            out_of_order = True
                        count += 1
                        if index in register_bits:
                            # If we have a bit in the position already it's been
                            # added by an earlier register in the circuit
                            # otherwise it's invalid qpy
                            if not in_circuit:
                                continue
                            raise exceptions.QpyError("Duplicate register bits found")

                        register_bits.add(index)

                    num_reg_bits = len(indices)
                    # Create a standlone register of the appropriate length (from
                    # the number of indices in the qpy data) and add it to the circuit
                    reg = reg_type(num_reg_bits, register_name)
                    # If any bits from qreg are out of order in the circuit handle
                    # is case
                    if out_of_order or not in_circuit:
                        for index, pos in sorted(
                            enumerate(x for x in indices if x >= 0), key=lambda x: x[1]
                        ):
                            if bit_type_label == "q":
                                bit_len = len(circ.qubits)
                            else:
                                bit_len = len(circ.clbits)
                            if pos < bit_len:
                                # If we have a bit in the position already it's been
                                # added by an earlier register in the circuit
                                # otherwise it's invalid qpy
                                if not in_circuit:
                                    continue
                                raise exceptions.QpyError("Duplicate register bits found")
                            # Fill any holes between the current register bit and the
                            # next one
                            if pos > bit_len:
                                bits = [bit_type() for _ in range(pos - bit_len)]
                                circ.add_bits(bits)
                            circ.add_bits([reg[index]])
                        if in_circuit:
                            circ.add_register(reg)
                    else:
                        if bit_type_label == "q":
                            bit_len = len(circ.qubits)
                        else:
                            bit_len = len(circ.clbits)
                        # If there is a hole between the start of the register and the
                        # current bits and standalone bits to fill the gap.
                        if start > len(circ.qubits):
                            bits = [bit_type() for _ in range(start - bit_len)]
                            circ.add_bits(bit_len)
                        if in_circuit:
                            circ.add_register(reg)
                        out_registers[bit_type_label][register_name] = reg
                else:
                    for index in indices:
                        if bit_type_label == "q":
                            bit_len = len(circ.qubits)
                        else:
                            bit_len = len(circ.clbits)
                        # Add any missing bits
                        bits = [bit_type() for _ in range(index + 1 - bit_len)]
                        circ.add_bits(bits)
                        if index in register_bits:
                            raise exceptions.QpyError("Duplicate register bits found")
                        register_bits.add(index)
                    if bit_type_label == "q":
                        bits = [circ.qubits[i] for i in indices]
                    else:
                        bits = [circ.clbits[i] for i in indices]
                    reg = reg_type(name=register_name, bits=bits)
                    if in_circuit:
                        circ.add_register(reg)
                    out_registers[bit_type_label][register_name] = reg
        # If we don't have sufficient bits in the circuit after adding
        # all the registers add more bits to fill the circuit
        if len(circ.qubits) < num_qubits:
            qubits = [Qubit() for _ in range(num_qubits - len(circ.qubits))]
            circ.add_bits(qubits)
        if len(circ.clbits) < num_clbits:
            clbits = [Clbit() for _ in range(num_qubits - len(circ.clbits))]
            circ.add_bits(clbits)
    else:
        circ = QuantumCircuit(
            num_qubits,
            num_clbits,
            name=name,
            global_phase=global_phase,
            metadata=metadata,
        )
    custom_instructions = _read_custom_instructions(file_obj, version, vectors)
    for _instruction in range(num_instructions):
        _read_instruction(file_obj, circ, out_registers, custom_instructions, version, vectors)
    for vec_name, (vector, initialized_params) in vectors.items():
        if len(initialized_params) != len(vector):
            warnings.warn(
                f"The ParameterVector: '{vec_name}' is not fully identical to its "
                "pre-serialization state. Elements "
                f"{', '.join([str(x) for x in set(range(len(vector))) - initialized_params])} "
                "in the ParameterVector will be not equal to the pre-serialized ParameterVector "
                f"as they weren't used in the circuit: {circ.name}",
                UserWarning,
            )

    return circ
