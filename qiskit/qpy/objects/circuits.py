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

"""Read and write circuit and circuit instructions"""

import io
import json
import struct
from collections import namedtuple

import numpy as np

from qiskit import circuit as circuit_mod
from qiskit import extensions
from qiskit.circuit import library
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.exceptions import QiskitError
from qiskit.extensions import quantum_initializer
from qiskit.qpy.common import (
    write_binary,
    read_binary,
    assign_key,
    TypeKey,
)
from .parameter_values import (
    dumps_parameter_value,
    loads_parameter_value,
)


# HEADER binary format
HEADER_V2 = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase_type",
        "global_phase_size",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
HEADER_V2_PACK = "!H1cHIIQIQ"
HEADER_V2_SIZE = struct.calcsize(HEADER_V2_PACK)

HEADER = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
HEADER_PACK = "!HdIIQIQ"
HEADER_SIZE = struct.calcsize(HEADER_PACK)

# CUSTOM_DEFINITIONS
# CUSTOM DEFINITION HEADER
CUSTOM_DEFINITION_HEADER = namedtuple("CUSTOM_DEFINITION_HEADER", ["size"])
CUSTOM_DEFINITION_HEADER_PACK = "!Q"
CUSTOM_DEFINITION_HEADER_SIZE = struct.calcsize(CUSTOM_DEFINITION_HEADER_PACK)

# CUSTOM_DEFINITION
CUSTOM_DEFINITION = namedtuple(
    "CUSTOM_DEFINITON",
    ["gate_name_size", "type", "num_qubits", "num_clbits", "custom_definition", "size"],
)
CUSTOM_DEFINITION_PACK = "!H1cII?Q"
CUSTOM_DEFINITION_SIZE = struct.calcsize(CUSTOM_DEFINITION_PACK)


# REGISTER binary format
REGISTER = namedtuple("REGISTER", ["type", "standalone", "size", "name_size"])
REGISTER_PACK = "!1c?IH"
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# INSTRUCTION binary format
INSTRUCTION = namedtuple(
    "INSTRUCTION",
    [
        "name_size",
        "label_size",
        "num_parameters",
        "num_qargs",
        "num_cargs",
        "has_condition",
        "condition_register_size",
        "value",
    ],
)
INSTRUCTION_PACK = "!HHHII?Hq"
INSTRUCTION_SIZE = struct.calcsize(INSTRUCTION_PACK)

# Instruction argument format
INSTRUCTION_ARG = namedtuple("INSTRUCTION_ARG", ["type", "size"])
INSTRUCTION_ARG_PACK = "!1cI"
INSTRUCTION_ARG_SIZE = struct.calcsize(INSTRUCTION_ARG_PACK)


def _read_header_v2(file_obj):
    header_raw = struct.unpack(HEADER_V2_PACK, file_obj.read(HEADER_V2_SIZE))
    header_tuple = HEADER_V2._make(header_raw)
    name = file_obj.read(header_tuple[0]).decode("utf8")

    # load global phase
    type_key = TypeKey(header_tuple[1].decode("utf8"))
    data = file_obj.read(header_tuple[2])
    if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
        global_phase = loads_parameter_value(type_key, data)
    else:
        raise TypeError("Invalid global phase type: %s" % type_key)

    header = {
        "global_phase": global_phase,
        "num_qubits": header_tuple[3],
        "num_clbits": header_tuple[4],
        "num_registers": header_tuple[6],
        "num_instructions": header_tuple[7],
    }
    metadata_raw = file_obj.read(header_tuple[5])
    metadata = json.loads(metadata_raw)
    return header, name, metadata


def _read_header(file_obj):
    header_raw = struct.unpack(HEADER_PACK, file_obj.read(HEADER_SIZE))
    header_tuple = HEADER._make(header_raw)
    name = file_obj.read(header_tuple[0]).decode("utf8")
    header = {
        "global_phase": header_tuple[1],
        "num_qubits": header_tuple[2],
        "num_clbits": header_tuple[3],
        "num_registers": header_tuple[5],
        "num_instructions": header_tuple[6],
    }
    metadata_raw = file_obj.read(header_tuple[4])
    metadata = json.loads(metadata_raw)
    return header, name, metadata


def _read_registers(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        register_raw = file_obj.read(REGISTER_SIZE)
        register = struct.unpack(REGISTER_PACK, register_raw)
        name = file_obj.read(register[3]).decode("utf8")
        standalone = register[1]
        REGISTER_ARRAY_PACK = "!%sI" % register[2]
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if register[0].decode("utf8") == "q":
            registers["q"][name] = (standalone, bit_indices)
        else:
            registers["c"][name] = (standalone, bit_indices)
    return registers


def _read_instruction(file_obj, circuit, registers, custom_instructions):
    instruction_raw = file_obj.read(INSTRUCTION_SIZE)
    instruction = struct.unpack(INSTRUCTION_PACK, instruction_raw)
    name_size = instruction[0]
    label_size = instruction[1]
    qargs = []
    cargs = []
    params = []
    gate_name = file_obj.read(name_size).decode("utf8")
    label = file_obj.read(label_size).decode("utf8")
    num_qargs = instruction[3]
    num_cargs = instruction[4]
    num_params = instruction[2]
    has_condition = instruction[5]
    register_name_size = instruction[6]
    condition_register = file_obj.read(register_name_size).decode("utf8")
    condition_value = instruction[7]
    condition_tuple = None
    if has_condition:
        # If an invalid register name is used assume it's a single bit
        # condition and treat the register name as a string of the clbit index
        if ClassicalRegister.name_format.match(condition_register) is None:
            # If invalid register prefixed with null character it's a clbit
            # index for single bit condition
            if condition_register[0] == "\x00":
                conditional_bit = int(condition_register[1:])
                condition_tuple = (circuit.clbits[conditional_bit], condition_value)
            else:
                raise ValueError(
                    f"Invalid register name: {condition_register} for condition register of "
                    f"instruction: {gate_name}"
                )
        else:
            condition_tuple = (registers["c"][condition_register], condition_value)
    qubit_indices = dict(enumerate(circuit.qubits))
    clbit_indices = dict(enumerate(circuit.clbits))

    # Load Arguments
    for _qarg in range(num_qargs):
        qarg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        qarg = struct.unpack(INSTRUCTION_ARG_PACK, qarg_raw)
        if qarg[0].decode("utf8") == "c":
            raise TypeError("Invalid input carg prior to all qargs")
        qargs.append(qubit_indices[qarg[1]])
    for _carg in range(num_cargs):
        carg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        carg = struct.unpack(INSTRUCTION_ARG_PACK, carg_raw)
        if carg[0].decode("utf8") == "q":
            raise TypeError("Invalid input qarg after all qargs")
        cargs.append(clbit_indices[carg[1]])

    # Load Parameters
    for _ in range(num_params):
        type_key, data = read_binary(file_obj)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            value = loads_parameter_value(type_key, data)
        elif type_key == TypeKey.STRING:
            value = data.decode("utf8")
        else:
            raise TypeError("Invalid parameter type: %s" % type_key)
        params.append(value)

    # Load Gate object
    gate_class = None
    if gate_name in ("Gate", "Instruction"):
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if label_size > 0:
            inst_obj.label = label
        circuit._append(inst_obj, qargs, cargs)
        return
    elif gate_name in custom_instructions:
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if label_size > 0:
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
    else:
        raise AttributeError("Invalid instruction type: %s" % gate_name)
    if gate_name == "Initialize":
        gate = gate_class(params)
    else:
        if gate_name == "Barrier":
            params = [len(qargs)]
        gate = gate_class(*params)
    gate.condition = condition_tuple
    if label_size > 0:
        gate.label = label
    if not isinstance(gate, Instruction):
        circuit.append(gate, qargs, cargs)
    else:
        circuit._append(gate, qargs, cargs)


def _parse_custom_instruction(custom_instructions, gate_name, params):
    (type_str, num_qubits, num_clbits, definition) = custom_instructions[gate_name]
    if type_str == "i":
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition:
            inst_obj.definition = definition
    elif type_str == "g":
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
    else:
        raise ValueError("Invalid custom instruction type '%s'" % type_str)
    return inst_obj


def _read_custom_instructions(file_obj, version):
    custom_instructions = {}
    custom_definition_header_raw = file_obj.read(CUSTOM_DEFINITION_HEADER_SIZE)
    custom_definition_header = struct.unpack(
        CUSTOM_DEFINITION_HEADER_PACK, custom_definition_header_raw
    )
    if custom_definition_header[0] > 0:
        for _ in range(custom_definition_header[0]):
            custom_definition_raw = file_obj.read(CUSTOM_DEFINITION_SIZE)
            custom_definition = struct.unpack(CUSTOM_DEFINITION_PACK, custom_definition_raw)
            (
                name_size,
                type_str,
                num_qubits,
                num_clbits,
                has_custom_definition,
                size,
            ) = custom_definition
            name = file_obj.read(name_size).decode("utf8")
            type_str = type_str.decode("utf8")
            definition_circuit = None
            if has_custom_definition:
                definition_buffer = io.BytesIO(file_obj.read(size))
                definition_circuit = read_circuit(definition_buffer, version)
            custom_instructions[name] = (type_str, num_qubits, num_clbits, definition_circuit)
    return custom_instructions


def _write_instruction(file_obj, instruction_tuple, custom_instructions, index_map):
    gate_class_name = instruction_tuple[0].__class__.__name__

    # pylint: disable=too-many-boolean-expressions
    if (
        (
            not hasattr(library, gate_class_name)
            and not hasattr(circuit_mod, gate_class_name)
            and not hasattr(extensions, gate_class_name)
            and not hasattr(quantum_initializer, gate_class_name)
        )
        or gate_class_name == "Gate"
        or gate_class_name == "Instruction"
        or isinstance(instruction_tuple[0], (library.BlueprintCircuit, library.PauliEvolutionGate))
    ):
        if instruction_tuple[0].name not in custom_instructions:
            custom_instructions[instruction_tuple[0].name] = instruction_tuple[0]
        gate_class_name = instruction_tuple[0].name

    has_condition = False
    condition_register = b""
    condition_value = 0
    if instruction_tuple[0].condition:
        has_condition = True
        if isinstance(instruction_tuple[0].condition[0], Clbit):
            bit_index = index_map["c"][instruction_tuple[0].condition[0]]
            condition_register = b"\x00" + str(bit_index).encode("utf8")
            condition_value = int(instruction_tuple[0].condition[1])
        else:
            condition_register = instruction_tuple[0].condition[0].name.encode("utf8")
            condition_value = instruction_tuple[0].condition[1]

    gate_class_name = gate_class_name.encode("utf8")
    label = getattr(instruction_tuple[0], "label")
    if label:
        label_raw = label.encode("utf8")
    else:
        label_raw = b""
    instruction_raw = struct.pack(
        INSTRUCTION_PACK,
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
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, b"q", index_map["q"][qbit])
        file_obj.write(instruction_arg_raw)
    for clbit in instruction_tuple[2]:
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, b"c", index_map["c"][clbit])
        file_obj.write(instruction_arg_raw)

    # Encode instruction params
    for param in instruction_tuple[0].params:
        type_key = assign_key(param)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            data = dumps_parameter_value(type_key, param)
        elif type_key == TypeKey.STRING:
            data = param.encode("utf8")
        else:
            raise TypeError(
                f"Invalid parameter type {instruction_tuple[0]} for gate {type(param)},"
            )

        write_binary(file_obj, data, type_key)


def _write_custom_instruction(file_obj, name, instruction):
    if isinstance(instruction, Gate):
        type_str = b"g"
    else:
        type_str = b"i"
    has_definition = False
    size = 0
    data = None
    num_qubits = instruction.num_qubits
    num_clbits = instruction.num_clbits
    if instruction.definition:
        has_definition = True
        definition_buffer = io.BytesIO()
        write_circuit(definition_buffer, instruction.definition)
        definition_buffer.seek(0)
        data = definition_buffer.read()
        definition_buffer.close()
        size = len(data)
    name_raw = name.encode("utf8")
    custom_instruction_raw = struct.pack(
        CUSTOM_DEFINITION_PACK,
        len(name_raw),
        type_str,
        num_qubits,
        num_clbits,
        has_definition,
        size,
    )
    file_obj.write(custom_instruction_raw)
    file_obj.write(name_raw)
    if data:
        file_obj.write(data)


def write_circuit(file_obj, circuit):
    """Write a single quantum circuit to the file like object.

    Args:
        file_obj (File): A file like object to write quantum circuit data.
        circuit (QuantumCircuit): A circuit program to write.

    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    metadata_raw = json.dumps(circuit.metadata, separators=(",", ":")).encode("utf8")
    metadata_size = len(metadata_raw)
    num_registers = len(circuit.qregs) + len(circuit.cregs)
    num_instructions = len(circuit)
    circuit_name = circuit.name.encode("utf8")

    type_key = assign_key(circuit.global_phase)
    if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
        global_phase_data = dumps_parameter_value(type_key, circuit.global_phase)
        global_phase_type = type_key.encode("utf8")
    else:
        raise TypeError("unsupported global phase type %s" % type(circuit.global_phase))

    header_raw = HEADER_V2(
        name_size=len(circuit_name),
        global_phase_type=global_phase_type,
        global_phase_size=len(global_phase_data),
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        metadata_size=metadata_size,
        num_registers=num_registers,
        num_instructions=num_instructions,
    )
    header = struct.pack(HEADER_V2_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(global_phase_data)
    file_obj.write(metadata_raw)
    qubit_indices = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: index for index, bit in enumerate(circuit.clbits)}
    if num_registers > 0:
        for reg in circuit.qregs:
            standalone = all(bit._register is reg for bit in reg)
            reg_name = reg.name.encode("utf8")
            file_obj.write(struct.pack(REGISTER_PACK, b"q", standalone, reg.size, len(reg_name)))
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = "!%sI" % reg.size
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *(qubit_indices[bit] for bit in reg)))
        for reg in circuit.cregs:
            standalone = all(bit._register is reg for bit in reg)
            reg_name = reg.name.encode("utf8")
            file_obj.write(struct.pack(REGISTER_PACK, b"c", standalone, reg.size, len(reg_name)))
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = "!%sI" % reg.size
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *(clbit_indices[bit] for bit in reg)))
    instruction_buffer = io.BytesIO()
    custom_instructions = {}
    index_map = {}
    index_map["q"] = qubit_indices
    index_map["c"] = clbit_indices
    for instruction in circuit.data:
        _write_instruction(instruction_buffer, instruction, custom_instructions, index_map)
    file_obj.write(struct.pack(CUSTOM_DEFINITION_HEADER_PACK, len(custom_instructions)))

    for name, instruction in custom_instructions.items():
        _write_custom_instruction(file_obj, name, instruction)

    instruction_buffer.seek(0)
    file_obj.write(instruction_buffer.read())
    instruction_buffer.close()


def read_circuit(file_obj, version):
    """Read a single quantum circuit from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        version (int): QPY version.

    Returns:
        QuantumCircuit: Deserialized quantum circuit object.

    Raises:
        TypeError: If any of the instructions is invalid data format.
        QiskitError: Invalid register index.
    """
    if version < 2:
        header, name, metadata = _read_header(file_obj)
    else:
        header, name, metadata = _read_header_v2(file_obj)

    global_phase = header["global_phase"]
    num_qubits = header["num_qubits"]
    num_clbits = header["num_clbits"]
    num_registers = header["num_registers"]
    num_instructions = header["num_instructions"]
    out_registers = {"q": {}, "c": {}}
    if num_registers > 0:
        circ = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        registers = _read_registers(file_obj, num_registers)

        for bit_type_label, bit_type, reg_type in [
            ("q", Qubit, QuantumRegister),
            ("c", Clbit, ClassicalRegister),
        ]:
            register_bits = set()
            # Add quantum registers and bits
            for register_name in registers[bit_type_label]:
                standalone, indices = registers[bit_type_label][register_name]
                if standalone:
                    start = min(indices)
                    count = start
                    out_of_order = False
                    for index in indices:
                        if not out_of_order and index != count:
                            out_of_order = True
                        count += 1
                        if index in register_bits:
                            raise QiskitError("Duplicate register bits found")
                        register_bits.add(index)

                    num_reg_bits = len(indices)
                    # Create a standlone register of the appropriate length (from
                    # the number of indices in the qpy data) and add it to the circuit
                    reg = reg_type(num_reg_bits, register_name)
                    # If any bits from qreg are out of order in the circuit handle
                    # is case
                    if out_of_order:
                        sorted_indices = np.argsort(indices)
                        for index in sorted_indices:
                            pos = indices[index]
                            if bit_type_label == "q":
                                bit_len = len(circ.qubits)
                            else:
                                bit_len = len(circ.clbits)
                            # Fill any holes between the current register bit and the
                            # next one
                            if pos > bit_len:
                                bits = [bit_type() for _ in range(pos - bit_len)]
                                circ.add_bits(bits)
                            circ.add_bits([reg[index]])
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
                            raise QiskitError("Duplicate register bits found")
                        register_bits.add(index)
                    if bit_type_label == "q":
                        bits = [circ.qubits[i] for i in indices]
                    else:
                        bits = [circ.clbits[i] for i in indices]
                    reg = reg_type(name=register_name, bits=bits)
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
    custom_instructions = _read_custom_instructions(file_obj, version)
    for _instruction in range(num_instructions):
        _read_instruction(file_obj, circ, out_registers, custom_instructions)

    return circ
