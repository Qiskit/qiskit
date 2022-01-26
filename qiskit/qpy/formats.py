# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Binary format definition."""

import struct
from collections import namedtuple


# FILE_HEADER
FILE_HEADER = namedtuple(
    "FILE_HEADER",
    ["preface", "qpy_version", "major_version", "minor_version", "patch_version", "num_circuits"],
)
FILE_HEADER_PACK = "!6sBBBBQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_PACK)

# CIRCUIT_HEADER_V2
CIRCUIT_HEADER_V2 = namedtuple(
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
CIRCUIT_HEADER_V2_PACK = "!H1cHIIQIQ"
CIRCUIT_HEADER_V2_SIZE = struct.calcsize(CIRCUIT_HEADER_V2_PACK)

# CIRCUIT_HEADER
CIRCUIT_HEADER = namedtuple(
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
CIRCUIT_HEADER_PACK = "!HdIIQIQ"
CIRCUIT_HEADER_SIZE = struct.calcsize(CIRCUIT_HEADER_PACK)

# REGISTER
REGISTER_V4 = namedtuple("REGISTER", ["type", "standalone", "size", "name_size", "in_circuit"])
REGISTER_V4_PACK = "!1c?IH?"
REGISTER_V4_SIZE = struct.calcsize(REGISTER_V4_PACK)

REGISTER = namedtuple("REGISTER", ["type", "standalone", "size", "name_size"])
REGISTER_PACK = "!1c?IH"
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# CIRCUIT_INSTRUCTION
CIRCUIT_INSTRUCTION = namedtuple(
    "CIRCUIT_INSTRUCTION",
    [
        "name_size",
        "label_size",
        "num_parameters",
        "num_qargs",
        "num_cargs",
        "has_condition",
        "condition_register_size",
        "condition_value",
    ],
)
CIRCUIT_INSTRUCTION_PACK = "!HHHII?Hq"
CIRCUIT_INSTRUCTION_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_PACK)

# CIRCUIT_INSTRUCTION_ARG
CIRCUIT_INSTRUCTION_ARG = namedtuple("CIRCUIT_INSTRUCTION_ARG", ["type", "size"])
CIRCUIT_INSTRUCTION_ARG_PACK = "!1cI"
CIRCUIT_INSTRUCTION_ARG_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_ARG_PACK)

# SparsePauliOp List
SPARSE_PAULI_OP_LIST_ELEM = namedtuple("SPARSE_PAULI_OP_LIST_ELEMENT", ["size"])
SPARSE_PAULI_OP_LIST_ELEM_PACK = "!Q"
SPARSE_PAULI_OP_LIST_ELEM_SIZE = struct.calcsize(SPARSE_PAULI_OP_LIST_ELEM_PACK)

# Pauli Evolution Gate
PAULI_EVOLUTION_DEF = namedtuple(
    "PAULI_EVOLUTION_DEF",
    ["operator_size", "standalone_op", "time_type", "time_size", "synth_method_size"],
)
PAULI_EVOLUTION_DEF_PACK = "!Q?1cQQ"
PAULI_EVOLUTION_DEF_SIZE = struct.calcsize(PAULI_EVOLUTION_DEF_PACK)

# CUSTOM_CIRCUIT_DEF_HEADER
CUSTOM_CIRCUIT_DEF_HEADER = namedtuple("CUSTOM_CIRCUIT_DEF_HEADER", ["size"])
CUSTOM_CIRCUIT_DEF_HEADER_PACK = "!Q"
CUSTOM_CIRCUIT_DEF_HEADER_SIZE = struct.calcsize(CUSTOM_CIRCUIT_DEF_HEADER_PACK)

# CUSTOM_CIRCUIT_INST_DEF
CUSTOM_CIRCUIT_INST_DEF = namedtuple(
    "CUSTOM_CIRCUIT_INST_DEF",
    ["gate_name_size", "type", "num_qubits", "num_clbits", "custom_definition", "size"],
)
CUSTOM_CIRCUIT_INST_DEF_PACK = "!H1cII?Q"
CUSTOM_CIRCUIT_INST_DEF_SIZE = struct.calcsize(CUSTOM_CIRCUIT_INST_DEF_PACK)

# INSTRUCTION_PARAM
INSTRUCTION_PARAM = namedtuple("TYPED_OBJECT", ["type", "size"])
INSTRUCTION_PARAM_PACK = "!1cQ"
INSTRUCTION_PARAM_SIZE = struct.calcsize(INSTRUCTION_PARAM_PACK)

# PARAMETER
PARAMETER = namedtuple("PARAMETER", ["name_size", "uuid"])
PARAMETER_PACK = "!H16s"
PARAMETER_SIZE = struct.calcsize(PARAMETER_PACK)

# COMPLEX
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"
COMPLEX_SIZE = struct.calcsize(COMPLEX_PACK)

# PARAMETER_VECTOR_ELEMENT
PARAMETER_VECTOR_ELEMENT = namedtuple(
    "PARAMETER_VECTOR_ELEMENT", ["vector_name_size", "vector_size", "uuid", "index"]
)
PARAMETER_VECTOR_ELEMENT_PACK = "!HQ16sQ"
PARAMETER_VECTOR_ELEMENT_SIZE = struct.calcsize(PARAMETER_VECTOR_ELEMENT_PACK)

# PARAMETER_EXPR
PARAMETER_EXPR = namedtuple("PARAMETER_EXPR", ["map_elements", "expr_size"])
PARAMETER_EXPR_PACK = "!QQ"
PARAMETER_EXPR_SIZE = struct.calcsize(PARAMETER_EXPR_PACK)

# PARAMETER_EXPR_MAP_ELEM_V3
PARAM_EXPR_MAP_ELEM_V3 = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["symbol_type", "type", "size"])
PARAM_EXPR_MAP_ELEM_V3_PACK = "!ccQ"
PARAM_EXPR_MAP_ELEM_V3_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_V3_PACK)

# PARAMETER_EXPR_MAP_ELEM
PARAM_EXPR_MAP_ELEM = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["type", "size"])
PARAM_EXPR_MAP_ELEM_PACK = "!cQ"
PARAM_EXPR_MAP_ELEM_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK)

# RANGE
RANGE = namedtuple("RANGE", ["start", "stop", "step"])
RANGE_PACK = "!qqq"
RANGE_SIZE = struct.calcsize(RANGE_PACK)

# SEQUENCE
SEQUENCE = namedtuple("SEQUENCE", ["num_elements"])
SEQUENCE_PACK = "!Q"
SEQUENCE_SIZE = struct.calcsize(SEQUENCE_PACK)
