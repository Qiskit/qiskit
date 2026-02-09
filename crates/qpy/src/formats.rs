// This code is part of Qiskit.
//
// (C) Copyright IBM 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bytes::Bytes;
use crate::expr::{read_expression, write_expression};
use crate::params::ParameterType;
use crate::value::{
    BitType, CircuitInstructionType, ExpressionType, ExpressionVarDeclaration, ModifierType,
    QPYReadData, QPYWriteData, RegisterType, ValueType,
};
use binrw::{BinRead, BinResult, BinWrite, Endian, binread, binrw, binwrite};
use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;
use qiskit_circuit::classical::expr::Expr;
use std::io::{Read, Seek, Write};
use std::marker::PhantomData;

/// The overall structure of the QPY data

// For now, the top-level is one circuit, and python handles the complete QPY file
// Only QPY version 17 is currently supported

// the main file structure:
// 1) Header: Contains the global data such as name, number of qubits etc.
// 2) Standalone vars: Contains the qiskit_circuit::Var elements used in expressions
// 3) Annotation Headers: The annotation-related global data.
// 4) Custom instructions: List of custom gates used in the circuits, e.g. gate with nonstandard control
// 5) Instruction: The sequential list of gates in the circuit.
// 6) Calibrations: Obsolete; this was pulse-related data. Here for backwards compatability.
// 7) Layout: The transpilation layout, if one exists (otherwise a dummy is used).
#[derive(BinWrite, Debug)]
#[brw(big)]
pub struct QPYCircuitV17 {
    pub header: CircuitHeaderV12Pack,
    pub standalone_vars: Vec<ExpressionVarDeclarationPack>,
    pub annotation_headers: AnnotationHeaderStaticPack,
    pub custom_instructions: CustomCircuitInstructionsPack,
    pub instructions: Vec<CircuitInstructionV2Pack>,
    pub calibrations: CalibrationsPack,
    pub layout: LayoutV2Pack,
}

// The header contains the global data of the circuit: name, global phase;
// number of qubits, clbits, instructions and vars;
// register data, metadata (as serialized bytes)
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CircuitHeaderV12Pack {
    #[bw(calc = circuit_name.len() as u16)]
    pub name_size: u16,
    pub global_phase_type: ValueType,
    #[bw(calc = global_phase_data.len() as u16)]
    pub global_phase_size: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    #[bw(calc = metadata.len() as u64)]
    pub metadata_size: u64,
    #[bw(calc = registers.len() as u32)]
    pub num_registers: u32,
    pub num_instructions: u64,
    pub num_vars: u32,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub circuit_name: String,
    #[br(count = global_phase_size as usize)]
    pub global_phase_data: Bytes,
    #[br(count = metadata_size)]
    pub metadata: Bytes,
    #[br(count = num_registers)]
    pub registers: Vec<RegisterV4Pack>,
}

// The data for a specific instruction in the circuit
// Each instruction has a name, an optional label,
// number of qubits ("qargs") and clbits ("cargs")
// and a "gate_class_name" used to identify the instruction (for Python-based gates, this will be the
// actual python class name, but for gates with native rust implementation this needs not be the case)
// many gates have parameters, stored as a vector of generic data since they take on various forms
// some gates are *controlled* (meaning they are applied only if the control qubits have a specific value)
// num_ctrl_qubits stores the number of qubits used in the control, and ctrl_state stores their expected value
// some gates have a **condition** (e.g. if-else instructions)
// some gates have **annotations** (e.g. box instruction)
// the "extras_key" is used to denote whether conditions and annotations are present
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[br(import(read_bits: bool))]
pub struct CircuitInstructionV2Pack {
    #[bw(calc = gate_class_name.len() as u16)]
    pub name_size: u16,
    #[bw(calc = label.len() as u16)]
    pub label_size: u16,
    #[bw(calc = params.len() as u16)]
    pub num_parameters: u16,
    pub num_qargs: u32,
    pub num_cargs: u32,
    pub extras_key: u8,
    #[bw(calc = condition.register_size)]
    pub condition_register_size: u16,
    #[bw(calc = condition.value)]
    pub condition_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub gate_class_name: String,
    #[br(count = label_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub label: String,
    #[br(parse_with = ConditionPack::read, args(condition_register_size, extract_conditional_key(extras_key), condition_value))]
    #[bw(write_with = ConditionPack::write)]
    pub condition: ConditionPack,
    #[br(count = num_qargs + num_cargs)]
    #[br(if(read_bits))]
    pub bit_data: Vec<CircuitInstructionArgPack>,
    #[br(count = num_parameters as usize)]
    pub params: Vec<GenericDataPack>,
    #[br(if(has_annotations(extras_key)))]
    pub annotations: Option<InstructionsAnnotationPack>,
}

// To save space, the extras key encoded data about the existance of annotations
// in its msb, and about the type of condition (Two-tuple, Expression or None) in the two lsbs.
pub mod extras_key_parts {
    pub const ANNOTATIONS: u8 = 0b1000_0000;
    pub const CONDITIONAL: u8 = 0b0000_0011;
}

// TODO: It may be possible to create two virtual fields, annotations_key and conditional_key
// such that they are set by the code and have binrw generate extras_key from them
fn has_annotations(extras_key: u8) -> bool {
    extras_key & extras_key_parts::ANNOTATIONS != 0
}

fn extract_conditional_key(extras_key: u8) -> u8 {
    extras_key & extras_key_parts::CONDITIONAL
}

// This struct is used in the mapping of a qubit/clbit appearing in an instruction to that qubit/clbit in the circuit
// each bit is identified by its type (Qubit or Clbit) and has its index in the circuit's qubit/clbit list
#[derive(BinWrite, BinRead)]
#[brw(big)]
#[derive(Debug)]
pub struct CircuitInstructionArgPack {
    pub bit_type: BitType,
    pub index: u32,
}

// a simple wrapper struct around the list of custom instructions, storing its length
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionsPack {
    #[bw(calc = custom_instructions.len() as u64)]
    pub custom_operations_length: u64,
    #[br(count = custom_operations_length)]
    pub custom_instructions: Vec<CustomCircuitInstructionDefPack>,
}

// A custom instruction definition. This is not a specific instantiation of the instruction
// But rather its global definition, used by the instructions in `CircuitInstructionV2Pack`
// so data fields like `bit_data` are not present. However, most of the usual "global" data is present.
// In addition, the `data` field are used to store serialized data relevant to the gate
// (e.g. python circuits or pauli evolution gate data) and the `base_gate` field is used
// to store the serialization of the instructions `base_gate`, if it exists.

// This struct seems more suitable as enum with binrw magic numbers
// unfortunately, since the gate_type field is second, binrw magic won't work here (TODO: fix in QPY18?)
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionDefPack {
    #[bw(calc = name.len() as u16)]
    pub gate_name_size: u16,
    pub gate_type: CircuitInstructionType,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub custom_definition: u8,
    #[bw(calc = data.len() as u64)]
    pub size: u64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    #[bw(calc = base_gate_raw.len() as u64)]
    pub base_gate_size: u64,
    #[br(count = gate_name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
    #[br(count = size)]
    pub data: Bytes,
    #[br(count = base_gate_size)]
    pub base_gate_raw: Bytes,
}

// Register data. Containing its type (qubits/clbits), its size, name,
// whether it's a standalone register or aliasing register,
// whether it's part of the circuit or not
// and for each bit present in the register, its index in the circuit
#[binread]
#[binwrite]
#[brw(big)]
#[derive(Debug)]
pub struct RegisterV4Pack {
    pub register_type: RegisterType,
    pub standalone: u8,
    #[bw(calc = bit_indices.len() as u32)]
    pub size: u32,
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    pub in_circuit: u8,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
    #[br(count = size)]
    pub bit_indices: Vec<i64>,
}

// Conditions
// There are three types of conidtions which are all bundled in the same pack:
// 1) None.
// 2) Two-tuple: a tuple of the form (register, target) where the register value should be compared with the target.
// In this case the target is a python int, represented in rust as BigUInt, but in python qpy it was saved using i64 so we keep it for now.
// Note that we also use (clbit, bool_target) as two tuple, where the clbit is encoded using the "\x00" hack that can be seen in ParamRegisterValue
// 3) Expresssion
// In the two-tuple representation, the target value is stored in the `value` field and the number of bytes in the serialized registered are stored in the
// `register_size` fields. Both are unused in the other cases, making the packing and decoding of this struct rather non-uniform.

// The most natural thing is to encode conditions as enum with magic numbers differentiating between different kinds
// however, since in QPY the condition data is not stored consecutively, it's going to be a mass to implement, so we do something else
// TODO: fix in QPY18?
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ConditionType {
    None = 0,
    TwoTuple = 1,
    Expression = 2,
}

impl From<u8> for ConditionType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::TwoTuple,
            2 => Self::Expression,
            _ => panic!("Invalid condition type specified {value}"),
        }
    }
}

// register SHOULD be a string, but since we encode some registers starting with "\x00" they are rendered illegal
// we should probably change this in future versions to support magic numbers (TODO: change in QPY18?)
#[derive(Debug)]
pub enum ConditionData {
    None,
    Register(Bytes),
    Expression(GenericDataPack),
}

// most of the data here is "virtual" in the sense that is is not stored as-is
// we use custom reader/writer to enable spreading the data to its relevant places in the qpy instruction
// in newer versions of qpy it may be better to store all the data consecutively
#[derive(Debug)]
pub struct ConditionPack {
    pub register_size: u16,
    pub value: i64,
    pub data: ConditionData,
}

impl Default for ConditionPack {
    fn default() -> Self {
        ConditionPack {
            register_size: 0u16,
            value: 0i64,
            data: ConditionData::None,
        }
    }
}

impl ConditionPack {
    pub(crate) fn key(&self) -> ConditionType {
        match self.data {
            ConditionData::Expression(_) => ConditionType::Expression,
            ConditionData::Register(_) => ConditionType::TwoTuple,
            ConditionData::None => ConditionType::None,
        }
    }

    pub(crate) fn write<W: Write + Seek>(
        value: &ConditionPack,
        writer: &mut W,
        endian: Endian,
        args: (),
    ) -> binrw::BinResult<()> {
        match &value.data {
            ConditionData::None => (),
            ConditionData::Register(register_data) => {
                register_data.write_options(writer, endian, args)?
            }
            ConditionData::Expression(expression_data) => {
                expression_data.write_options(writer, endian, args)?
            }
        }
        Ok(())
    }

    pub(crate) fn read<R: Read + Seek>(
        reader: &mut R,
        endian: Endian,
        (register_size, key, value): (u16, u8, i64),
    ) -> BinResult<ConditionPack> {
        let data = match ConditionType::from(key) {
            ConditionType::TwoTuple => {
                let mut buf = vec![0u8; register_size as usize];
                reader.read_exact(&mut buf)?;
                ConditionData::Register(buf.into())
            }
            ConditionType::Expression => {
                ConditionData::Expression(GenericDataPack::read_options(reader, endian, ())?)
            }
            ConditionType::None => ConditionData::None,
        };
        Ok(ConditionPack {
            register_size,
            value,
            data,
        })
    }
}

// Transpilation layout data, based on the Python version of TranspileLayout (now converted to Rust)
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct LayoutV2Pack {
    pub exists: u8,
    pub initial_layout_size: i32,
    pub input_mapping_size: i32,
    pub final_layout_size: i32,
    #[bw(calc = extra_registers.len() as u32)]
    pub extra_registers_length: u32,
    pub input_qubit_count: i32,
    #[br(count = extra_registers_length)]
    pub extra_registers: Vec<RegisterV4Pack>,
    #[br(count = initial_layout_size.max(0))]
    pub initial_layout_items: Vec<InitialLayoutItemV2Pack>,
    #[br(count = input_mapping_size.max(0))]
    pub input_mapping_items: Vec<u32>,
    #[br(count = final_layout_size.max(0))]
    pub final_layout_items: Vec<u32>,
}

// Data for initial layout item: its index and its register name, stored in a rather ad-hoc manner
// TODO: Improve in QPY18?
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InitialLayoutItemV2Pack {
    pub index_value: i32,
    pub register_name_length: i32, // in this special case, reg_name_length can be -1 indicating "no name"
    #[br(count = register_name_length.max(0) as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub register_name: String,
}

// A serialized "generic data".
// This can be a wide variety of values, mostly used in instruction parameters.
// Some (not all) examples:
// integers (e.g. for switch statements and loops),
// floats, compelx numbers (e.g. for rx gates),
// Parameters (and parameter expressions and vectors) (e.g. for rx gates),
// Expressions (e.g. for complex conditions in control flow statements),
// Circuits (e.g. for control flow statements having a body)
#[binread]
#[binwrite]
#[brw(big)]
#[derive(Debug)]
pub struct GenericDataPack {
    pub type_key: ValueType,
    #[bw(calc = data.len() as u64)]
    pub data_len: u64,
    #[br(count = data_len)]
    pub data: Bytes,
}

// A simple contained for a sequence of generic data items, containing its length.
#[binrw]
#[derive(Debug)]
#[brw(big)]
pub struct GenericDataSequencePack {
    #[bw(calc = elements.len() as u64)]
    pub num_elements: u64,
    #[br(count = num_elements)]
    pub elements: Vec<GenericDataPack>,
}

// the specific data required for a pauli evolution gate
//
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[br(import (version: u32))]
pub struct PauliEvolutionDefPack {
    #[bw(calc = pauli_data.len() as u64)]
    pub operator_size: u64,
    pub standalone_op: u8,
    pub time_type: ValueType,
    #[bw(calc = time_data.len() as u64)]
    pub time_size: u64,
    #[bw(calc = synth_data.len() as u64)]
    pub synth_method_size: u64,
    #[br(count = operator_size, args { inner: (version,) })]
    pub pauli_data: Vec<PauliDataPack>,
    #[br(count = time_size)]
    pub time_data: Bytes,
    #[br(count = synth_method_size)]
    pub synth_data: Bytes,
}

// A pauli operator data for pauli evolution gates
// The operator is given either as a SparesePauliOp list or as a SparasePauliObservable
// SparasePauliObservable was added in V17
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum PauliDataPackV17 {
    #[brw(magic = 0u8)] // old style: sparse pauli op list
    SparsePauliOp(SparsePauliOpListElemPack),
    #[brw(magic = 1u8)] // new style added in v17: sparse observable
    SparseObservable(SparsePauliObservableElemPack),
}

// The V16 version of the Pauli data pack only allows SparsePauliOp and doesn't use a distinguishing first byte
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum PauliDataPackV16 {
    SparsePauliOp(SparsePauliOpListElemPack),
}

#[binrw]
#[derive(Debug)]
#[br(import(version: u32))]
pub enum PauliDataPack {
    #[br(pre_assert(version <= 16))]
    V16(PauliDataPackV16),

    #[br(pre_assert(version >= 17))]
    V17(PauliDataPackV17),
}

// SparsePauliOpList is a serialized python numpy array
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct SparsePauliOpListElemPack {
    #[bw(calc = data.len() as u64)]
    pub size: u64,
    #[br(count = size)]
    pub data: Bytes,
}

// SparsePauiObservable has explicit data that can be used to reconstruct
// a rust SparseObservable struct
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct SparsePauliObservableElemPack {
    pub num_qubits: u32,
    #[bw(calc = coeff_data.len() as u64)]
    pub coeff_data_size: u64,
    #[bw(calc = bitterm_data.len() as u64)]
    pub bitterm_data_size: u64,
    #[bw(calc = inds_data.len() as u64)]
    pub inds_data_size: u64,
    #[bw(calc = bounds_data.len() as u64)]
    pub bounds_data_size: u64,
    #[br(count = coeff_data_size)]
    pub coeff_data: Vec<f64>, // complex numbers stored in format [re1, im1, re2, im2,...]
    #[br(count = bitterm_data_size)]
    pub bitterm_data: Vec<u16>,
    #[br(count = inds_data_size)]
    pub inds_data: Vec<u32>,
    #[br(count = bounds_data_size)]
    pub bounds_data: Vec<u64>,
}

// *****Parameter Expression handling (used in params.rs)*****

// A single parameter symbol, consisting of a name and uuid
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterSymbolPack {
    #[bw(calc = name.len() as u16)]
    pub name_length: u16,
    pub uuid: [u8; 16],
    #[br(count = name_length as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
}

// A single parameter vector element. Since vectors has no standalone representation in QPY
// the vector data (name and size) is stored along with the element-specific data (uuid and index in the vector)
// This is obviously not optimal compared to storing a list of vector and keeping a pointer in each element so TODO: improve in QPY18?
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterVectorElementPack {
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    pub vector_size: u64,
    pub uuid: [u8; 16],
    pub index: u64,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
}

// The various types of components available in a parameter expression
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum ParameterExpressionElementPack {
    #[brw(magic = 0u8)]
    Add(ParameterExpressionStandardOpPack),
    #[brw(magic = 1u8)]
    Sub(ParameterExpressionStandardOpPack),
    #[brw(magic = 2u8)]
    Mul(ParameterExpressionStandardOpPack),
    #[brw(magic = 3u8)]
    Div(ParameterExpressionStandardOpPack),
    #[brw(magic = 4u8)]
    Pow(ParameterExpressionStandardOpPack),
    #[brw(magic = 5u8)]
    Sin(ParameterExpressionStandardOpPack),
    #[brw(magic = 6u8)]
    Cos(ParameterExpressionStandardOpPack),
    #[brw(magic = 7u8)]
    Tan(ParameterExpressionStandardOpPack),
    #[brw(magic = 8u8)]
    Asin(ParameterExpressionStandardOpPack),
    #[brw(magic = 9u8)]
    Acos(ParameterExpressionStandardOpPack),
    #[brw(magic = 10u8)]
    Exp(ParameterExpressionStandardOpPack),
    #[brw(magic = 11u8)]
    Log(ParameterExpressionStandardOpPack),
    #[brw(magic = 12u8)]
    Sign(ParameterExpressionStandardOpPack),
    #[brw(magic = 13u8)]
    Grad(ParameterExpressionStandardOpPack),
    #[brw(magic = 14u8)]
    Conj(ParameterExpressionStandardOpPack),
    #[brw(magic = 15u8)]
    Substitute(ParameterExpressionSubsOpPack),
    #[brw(magic = 16u8)]
    Abs(ParameterExpressionStandardOpPack),
    #[brw(magic = 17u8)]
    Atan(ParameterExpressionStandardOpPack),
    #[brw(magic = 18u8)]
    Rsub(ParameterExpressionStandardOpPack),
    #[brw(magic = 19u8)]
    Rdiv(ParameterExpressionStandardOpPack),
    #[brw(magic = 20u8)]
    Rpow(ParameterExpressionStandardOpPack),
    #[brw(magic = 255u8)]
    Expression(ParameterExpressionStandardOpPack),
}

// A compact representation the operand data in a parameter expression node
// For each operand, stores its type (which can be null) and the uuid of the
// operand node (saving space and allowing for reuse of identical operands)
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub struct ParameterExpressionStandardOpPack {
    pub lhs_type: ParameterType,
    pub lhs: [u8; 16],
    pub rhs_type: ParameterType,
    pub rhs: [u8; 16],
}

// The data for a parameter expression substitution operand
// Which is somewhat different than the rest, since it contains a serialized
// mapping of symbols to parameter expressions. This is obsolete in qiskit 2.3
// and so might be removed in the future versions of QPY (TODO: fix it QPY18?)
#[binrw]
#[derive(Debug)]
#[brw(big)]
pub struct ParameterExpressionSubsOpPack {
    #[bw(calc = "u".as_bytes()[0])]
    pub _lhs_type: u8,
    #[bw(calc = {
        let mut size = [0u8; 16];
        let len_bytes = (mapping_data.len() as u64).to_be_bytes();
        size[..8].copy_from_slice(&len_bytes);
        size
    })]
    pub mapping_data_size: [u8; 16],
    #[bw(calc = "n".as_bytes()[0])]
    pub _rhs_type: u8,
    #[bw(calc = [0u8; 16])]
    pub _rhs: [u8; 16],

    #[br(count = u64::from_be_bytes(mapping_data_size[..8].try_into().unwrap()))]
    pub mapping_data: Bytes,
}

/// A Parameter Expression is stored in two chunks (along with length data)
/// 1) The parameter expression data itself, already serialized
/// 2) The symbol table for the parameter expression (to save space when using the same symbol more than once in the expression)
/// It's not completely clear to me why the symbol table was originally structrued as it was, since not all the data is used
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionPack {
    #[bw(calc = symbol_table_data.len() as u64)]
    pub symbol_tables_length: u64,
    #[bw(calc = expression_data.len() as u64)]
    pub expression_data_length: u64,
    #[br(count = expression_data_length)]
    pub expression_data: Bytes,
    #[br(count = symbol_tables_length)]
    pub symbol_table_data: Vec<ParameterExpressionSymbolPack>,
}

// This is the data type for the dictionary-like symbol table, storing both the parameter expression symbol
// (serving as key) and its corresponding value (given as a generic serialized value)
// It's not clear to me whether this can be optimised (TODO: check for QPY18) or if all the data here is required
// e.g. whether we can avoid storing the symbols and only use their uuid
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ParameterExpressionSymbolPack {
    #[brw(magic = b'p')]
    Parameter(ParameterExpressionParameterSymbolPack),
    #[brw(magic = b'v')]
    ParameterVector(ParameterExpressionParameterVectorSymbolPack),
    #[brw(magic = b'e')]
    ParameterExpression(ParameterExpressionParameterExpressionSymbolPack),
}

// symbol->value mapping for parameter expressions
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterSymbolPack {
    pub value_key: ValueType,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterSymbolPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

// vector symbol->value mapping for parameter expressions
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterVectorSymbolPack {
    pub value_key: ValueType,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterVectorElementPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

// parameter expression->value mapping for parameter expressions
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterExpressionSymbolPack {
    pub value_key: u8,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterExpressionPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

// Mappings are general dict-like mappings from serialized bytes to serialized bytes
// However, they are only used in Parameter Expression handling and we might be able to remove them completely
// Since we want to avoid storing serialized bytes in structs. So TODO: fix in QPY18?
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct MappingPack {
    #[bw(calc = items.len() as u64)]
    pub num_elements: u64,
    #[br(count = num_elements)]
    pub items: Vec<MappingItem>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct MappingItem {
    #[bw(calc = key_bytes.len() as u16)]
    pub key_size: u16,
    pub item_type: ValueType,
    #[bw(calc = item_bytes.len() as u16)]
    pub size: u16,
    #[br(count = usize::from(key_size))]
    pub key_bytes: Bytes,
    #[br(count = usize::from(size))]
    pub item_bytes: Bytes,
}

// *****Expression handling (used in expr.rs)*****
// expressions are stored as a consecutive list of expression elements
// that encode a tree structure in inorder traversal
// since QPY doesn't explicitly store the number of elements in the expression
// we need to manually handle byte-level reading along with parsing the expression
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[br(import(qpy_read_data: &'a QPYReadData<'a>))]
#[bw(import(qpy_write_data: &'a QPYWriteData<'a>))]
pub struct ExpressionPack<'a> {
    #[br(parse_with = read_expression, args(qpy_read_data))]
    #[bw(write_with = write_expression, args(qpy_write_data))]
    pub expression: Expr,

    #[br(ignore)]
    #[bw(ignore)]
    pub _phantom: PhantomData<&'a Option<Expr>>,
}

// The types of values for elements of the expression - boolean and specific-width ints
// That can correspond to the values of specific clbits, floats and durations
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum ExpressionTypePack {
    #[brw(magic = b'b')]
    Bool,
    #[brw(magic = b'u')]
    Int(u32), // int type also have a width parameter
    #[brw(magic = b'f')]
    Float,
    #[brw(magic = b'd')]
    Duration,
}

// The various node types in an expression:
// Either a variable, a stretch, a concrete value, a cast,
// a unary op, a binary op or an index.
// These correspond to qiskit_circuit::classical::expr::expr
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum ExpressionElementPack {
    #[brw(magic = b'x')]
    Var(ExpressionTypePack, ExpressionVarElementPack),
    #[brw(magic = b's')]
    Stretch(ExpressionTypePack, u16),
    #[brw(magic = b'v')]
    Value(ExpressionTypePack, ExpressionValueElementPack),
    #[brw(magic = b'c')]
    Cast(ExpressionTypePack, u8),
    #[brw(magic = b'u')]
    Unary(ExpressionTypePack, u8),
    #[brw(magic = b'b')]
    Binary(ExpressionTypePack, u8),
    #[brw(magic = b'i')]
    Index(ExpressionTypePack),
}

// An expression's var data - either a clbit, a registe, or given by a uuid (for a standalone var)
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum ExpressionVarElementPack {
    #[brw(magic = b'C')]
    Clbit(u32),
    #[brw(magic = b'R')]
    Register(ExpressionVarRegisterPack),
    #[brw(magic = b'U')]
    Uuid(u16), // the uuid in the standalone vars list
}

// An expression register data, specifically it's name
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ExpressionVarRegisterPack {
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
}

// The storage for a value of an expression value
// Note that integers are arbitrary large
// TODO: this may be consolidated with GenericValue in QPY18?
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum ExpressionValueElementPack {
    #[brw(magic = b'b')]
    Bool(u8),
    #[brw(magic = b'i')]
    Int(BigIntPack),
    #[brw(magic = b'f')]
    Float(f64),
    #[brw(magic = b't')]
    Duration(DurationPack),
}

// An enum for the various duration types and their values
#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub enum DurationPack {
    #[brw(magic = b't')] // DT
    DT(u64),
    #[brw(magic = b'p')] // PS
    PS(f64),
    #[brw(magic = b'n')] // NS
    NS(f64),
    #[brw(magic = b'u')] // US
    US(f64),
    #[brw(magic = b'm')] // MS
    MS(f64),
    #[brw(magic = b's')] // S
    S(f64),
}

// A struct for storing arbitrary-length integers, as they are saved in QPY
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct BigIntPack {
    #[bw(calc = bytes.len() as u8)]
    pub num_bytes: u8,
    #[br(count = num_bytes as usize)]
    pub bytes: Bytes,
}

// A struct for storing ranges
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct RangePack {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

// A struct for storing modifiers
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ModifierPack {
    pub modifier_type: ModifierType, // this is a u8
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    pub power: f64,
}

// This is a declaration of a variable that may appear iniside various
// Expressions in the circuit. It contains its uuid, name,
// usage type (input/capture/local/etc) and type (bool, int, float etc.)
// This data is not used directly in any expression; rather, its written to the "standalone vars"
// part of the qpy file and used when reconstructing expressions that use them (given by their uuid)
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ExpressionVarDeclarationPack {
    pub uuid_bytes: [u8; 16],
    pub usage: ExpressionVarDeclaration,
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    pub exp_type: ExpressionType,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
}

// Calibrations are obsolete and won't be used in Qiskit 2.0 and beyond, but we still need to consume that part of the qpy file
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CalibrationsPack {
    pub num_cals: u16,
    // TODO: incomplete; will be needed for previous versions support
}

// *****Annotation-related data types*****

// The index of the annotation's namespace, and its serialized payload
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InstructionAnnotationPack {
    pub namespace_index: u32,
    #[bw(calc = payload.len() as u64)]
    pub payload_size: u64,
    #[br(count = payload_size)]
    pub payload: Bytes,
}

// A list of annotations
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InstructionsAnnotationPack {
    #[bw(calc = annotations.len() as u32)]
    pub num_annotations: u32,
    #[br(count = num_annotations)]
    pub annotations: Vec<InstructionAnnotationPack>,
}

// A seriqlized annotation state, along with its namespace
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct AnnotationStateHeaderPack {
    #[bw(calc = namespace.len() as u32)]
    pub namespace_size: u32,
    #[bw(calc = state.len() as u64)]
    pub state_size: u64,
    #[br(count = namespace_size, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub namespace: String,
    #[br(count = state_size)]
    pub state: Bytes,
}

// A list of annotation state headers
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct AnnotationHeaderStaticPack {
    #[bw(calc = state_headers.len() as u32)]
    pub num_namespaces: u32,
    #[br(count = num_namespaces)]
    pub state_headers: Vec<AnnotationStateHeaderPack>,
}

// implementations of custom read/write for the more complex data types
impl BinRead for QPYCircuitV17 {
    type Args<'a> = ();

    fn read_options<R: Read + Seek>(
        reader: &mut R,
        endian: Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let header = CircuitHeaderV12Pack::read_options(reader, endian, ())?;
        let mut standalone_vars = Vec::with_capacity(header.num_vars as usize);
        for _ in 0..header.num_vars {
            standalone_vars.push(ExpressionVarDeclarationPack::read_options(
                reader,
                endian,
                (),
            )?);
        }
        let annotation_headers = AnnotationHeaderStaticPack::read_options(reader, endian, ())?;
        let custom_instructions = CustomCircuitInstructionsPack::read_options(reader, endian, ())?;
        let mut instructions = Vec::with_capacity(header.num_vars as usize);
        for _ in 0..header.num_instructions {
            // read instructions, including circuit bits (the `true` arg)
            instructions.push(CircuitInstructionV2Pack::read_options(
                reader,
                endian,
                (true,),
            )?);
        }
        let calibrations = CalibrationsPack::read_options(reader, endian, ())?;
        let layout = LayoutV2Pack::read_options(reader, endian, ())?;
        Ok(Self {
            header,
            standalone_vars,
            annotation_headers,
            custom_instructions,
            instructions,
            calibrations,
            layout,
        })
    }
}

/// helper for passing PyResult errors that arise during binrw parsing
pub fn to_binrw_error<W: Seek, E: std::error::Error + Send + Sync + 'static>(
    writer: &mut W,
    err: E,
) -> binrw::Error {
    binrw::Error::Custom {
        pos: writer.stream_position().unwrap_or(0),
        err: Box::new(err),
    }
}

/// helper for converting custom binrw errors back to PyResult
pub fn from_binrw_error(err: binrw::Error) -> PyErr {
    match err {
        binrw::Error::Custom { err, .. } => {
            if let Ok(qpy_err) = err.downcast::<PyErr>() {
                *qpy_err
            } else {
                PyRuntimeError::new_err("unknown error")
            }
        }
        _ => PyRuntimeError::new_err("unknown error"),
    }
}
