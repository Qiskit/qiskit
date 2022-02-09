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


# FILE_HEADER binary format
FILE_HEADER = namedtuple(
    "FILE_HEADER",
    ["preface", "qpy_version", "major_version", "minor_version", "patch_version", "num_circuits"],
)
FILE_HEADER_PACK = "!6sBBBBQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_PACK)

# CIRCUIT_HEADER_V2 binary format
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

# CIRCUIT_HEADER binary format
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

# REGISTER binary format
REGISTER = namedtuple("REGISTER", ["type", "standalone", "size", "name_size"])
REGISTER_PACK = "!1c?IH"
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# CIRCUIT_INSTRUCTION binary format
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

# CIRCUIT_INSTRUCTION_ARG binary format
CIRCUIT_INSTRUCTION_ARG = namedtuple("CIRCUIT_INSTRUCTION_ARG", ["type", "size"])
CIRCUIT_INSTRUCTION_ARG_PACK = "!1cI"
CIRCUIT_INSTRUCTION_ARG_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_ARG_PACK)

# SparsePauliOp List binary format
SPARSE_PAULI_OP_LIST_ELEM = namedtuple("SPARSE_PAULI_OP_LIST_ELEMENT", ["size"])
SPARSE_PAULI_OP_LIST_ELEM_PACK = "!Q"
SPARSE_PAULI_OP_LIST_ELEM_SIZE = struct.calcsize(SPARSE_PAULI_OP_LIST_ELEM_PACK)

# Pauli Evolution Gate binary format
PAULI_EVOLUTION_DEF = namedtuple(
    "PAULI_EVOLUTION_DEF",
    ["operator_size", "standalone_op", "time_type", "time_size", "synth_method_size"],
)
PAULI_EVOLUTION_DEF_PACK = "!Q?1cQQ"
PAULI_EVOLUTION_DEF_SIZE = struct.calcsize(PAULI_EVOLUTION_DEF_PACK)

# CUSTOM_CIRCUIT_DEF_HEADER binary format
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

# TYPED_OBJECT binary format
TYPED_OBJECT = namedtuple("TYPED_OBJECT", ["type", "size"])
TYPED_OBJECT_PACK = "!1cQ"
TYPED_OBJECT_PACK_SIZE = struct.calcsize(TYPED_OBJECT_PACK)

# PARAMETER binary format
PARAMETER = namedtuple("PARAMETER", ["name_size", "uuid"])
PARAMETER_PACK = "!H16s"
PARAMETER_SIZE = struct.calcsize(PARAMETER_PACK)

# COMPLEX binary format
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"
COMPLEX_PACK_SIZE = struct.calcsize(COMPLEX_PACK)

# PARAMETER_VECTOR_ELEMENT binary format
PARAMETER_VECTOR_ELEMENT = namedtuple(
    "PARAMETER_VECTOR_ELEMENT", ["vector_name_size", "vector_size", "uuid", "index"]
)
PARAMETER_VECTOR_ELEMENT_PACK = "!HQ16sQ"
PARAMETER_VECTOR_ELEMENT_SIZE = struct.calcsize(PARAMETER_VECTOR_ELEMENT_PACK)

# PARAMETER_EXPR binary format
PARAMETER_EXPR = namedtuple("PARAMETER_EXPR", ["map_elements", "expr_size"])
PARAMETER_EXPR_PACK = "!QQ"
PARAMETER_EXPR_SIZE = struct.calcsize(PARAMETER_EXPR_PACK)

# PARAMETER_EXPR_MAP_ELEM_V3 binary format
PARAM_EXPR_MAP_ELEM_V3 = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["symbol_type", "type", "size"])
PARAM_EXPR_MAP_ELEM_PACK_V3 = "!ccQ"
PARAM_EXPR_MAP_ELEM_SIZE_V3 = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK_V3)

# PARAMETER_EXPR_MAP_ELEM binary format
PARAM_EXPR_MAP_ELEM = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["type", "size"])
PARAM_EXPR_MAP_ELEM_PACK = "!cQ"
PARAM_EXPR_MAP_ELEM_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK)
