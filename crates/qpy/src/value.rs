// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
use binrw::meta::{ReadEndian, WriteEndian};
use binrw::{BinRead, BinWrite, Endian, binrw};
use hashbrown::HashMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyComplex, PyDict, PyFloat, PyInt};

use qiskit_circuit::bit::ShareableClbit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::classical::expr::{Expr, Stretch, Var};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::object_registry::ObjectRegistry;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::{Clbit, imports};

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::circuit_reader::unpack_circuit;
use crate::formats::{self, GenericDataPack, GenericDataSequencePack};
use crate::params::{pack_parameter_vector, pack_symbol, unpack_parameter_vector, unpack_symbol};
use crate::py_methods::{
    py_deserialize_range, py_unpack_generic_sequence_to_tuple, py_unpack_modifier,
};
use crate::{QpyError, UnsupportedFeatureForVersion};

use core::fmt;
use num_complex::Complex64;
use std::fmt::Debug;
use std::io::Cursor;
use uuid::Uuid;

pub const QPY_VERSION: u32 = 15;

// Standard char representation of register types: 'q' qreg, 'c' for creg
pub mod register_types {
    pub const QREG: u8 = b'q';
    pub const CREG: u8 = b'c';
}

pub mod bit_types {
    pub const QUBIT: u8 = b'q';
    pub const CLBIT: u8 = b'c';
}

#[derive(Debug)]
pub struct QPYWriteData<'a> {
    pub version: u32,
    pub _use_symengine: bool, // TODO: remove this field?
    pub clbits: &'a ObjectRegistry<Clbit, ShareableClbit>,
    pub standalone_var_indices: HashMap<u128, u16>, // mapping from the variable's UUID to its index in the standalone variables list
    pub annotation_handler: AnnotationHandler<'a>,
}
#[derive(Debug)]
pub struct QPYReadData<'a> {
    pub circuit_data: &'a mut CircuitData,
    pub version: u32,
    pub use_symengine: bool,
    pub cregs: Py<PyDict>,
    pub standalone_vars: HashMap<u16, qiskit_circuit::Var>,
    pub standalone_stretches: HashMap<u16, qiskit_circuit::Stretch>,
    pub vectors: HashMap<Uuid, (Py<PyAny>, Vec<u32>)>,
    pub annotation_handler: AnnotationHandler<'a>,
}

pub mod tags {
    pub const BOOL: u8 = b'b';
    pub const INTEGER: u8 = b'i';
    pub const FLOAT: u8 = b'f';
    pub const COMPLEX: u8 = b'c';
    pub const CASE_DEFAULT: u8 = b'd';
    pub const REGISTER: u8 = b'R';
    pub const RANGE: u8 = b'r';
    pub const TUPLE: u8 = b't';
    pub const NUMPY_OBJ: u8 = b'n';
    pub const PARAMETER: u8 = b'p';
    pub const PARAMETER_VECTOR: u8 = b'v';
    pub const PARAMETER_EXPRESSION: u8 = b'e';
    pub const STRING: u8 = b's';
    pub const NULL: u8 = b'z';
    pub const EXPRESSION: u8 = b'x';
    pub const MODIFIER: u8 = b'm';
    pub const CIRCUIT: u8 = b'q';
}

pub mod modifier_types {
    pub const INVERSE: u8 = b'i';
    pub const CONTROL: u8 = b'c';
    pub const POWER: u8 = b'p';
}

#[binrw]
#[derive(Debug)]
pub enum ExpressionType {
    #[brw(magic = b'b')]
    Bool,
    #[brw(magic = b'u')]
    Uint(u32),
    #[brw(magic = b'f')]
    Float,
    #[brw(magic = b'd')]
    Duration,
}

pub mod expression_var_declaration {
    pub const INPUT: u8 = b'I';
    pub const CAPTURE: u8 = b'C';
    pub const LOCAL: u8 = b'L';
    pub const STRETCH_CAPTURE: u8 = b'A';
    pub const STRETCH_LOCAL: u8 = b'O';
}

pub fn serialize<T>(value: &T) -> Bytes
where
    T: BinWrite + WriteEndian + Debug,
    for<'a> <T as BinWrite>::Args<'a>: Default,
{
    let mut buffer = Cursor::new(Vec::new());
    value.write(&mut buffer).unwrap();
    buffer.into()
}
pub fn serialize_with_args<T, A>(value: &T, args: A) -> Bytes
where
    T: BinWrite<Args<'static> = A> + WriteEndian + Debug,
    A: Clone + Debug,
{
    let mut buffer = Cursor::new(Vec::new());
    value.write_args(&mut buffer, args).unwrap();
    buffer.into()
}

pub fn deserialize<T>(bytes: &[u8]) -> PyResult<(T, usize)>
where
    T: BinRead<Args<'static> = ()> + Debug,
{
    let mut cursor = Cursor::new(bytes);
    let value = T::read_options(&mut cursor, Endian::Big, ()).unwrap();
    let bytes_read = cursor.position() as usize;
    Ok((value, bytes_read))
}

pub fn deserialize_with_args<T, A>(bytes: &[u8], args: A) -> PyResult<(T, usize)>
where
    T: BinRead<Args<'static> = A> + ReadEndian + Debug,
    A: Clone,
{
    let mut cursor = Cursor::new(bytes);
    let value = T::read_args(&mut cursor, args).unwrap();
    let bytes_read = cursor.position() as usize;
    Ok((value, bytes_read))
}

pub fn deserialize_vec<T>(mut bytes: &[u8]) -> PyResult<Vec<T>>
where
    T: BinRead<Args<'static> = ()> + Debug,
{
    let mut result = Vec::new();
    while !bytes.is_empty() {
        let (item, pos) = deserialize::<T>(bytes)?;
        result.push(item);
        bytes = &bytes[pos..];
    }
    Ok(result)
}

// An enum for all rust-space data that might be found in a serialized ("Bytes") form inside the qpy file
// Under ideal conditions, we won't need such an enum since every rust based value will be present explicitly
// In the formats.rs file. However, due to the legacy of how Python-based value were stored such explicit
// representation is not always possible or difficult to implement
// This is the pure rust alternative to `DumpedPyValue` which is used to serialize data given via python objects
#[derive(Clone, Debug)]
pub enum GenericValue {
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Complex64(Complex64),
    ParameterExpressionSymbol(Symbol),
    ParameterExpressionVectorSymbol(Symbol),
    Tuple(Vec<GenericValue>),
}

// we want to be able to extract the value relatively painlessly;
// e.g. let my_bool = value.as_typed::<bool>().unwrap()
pub trait FromGenericValue: Sized {
    fn from_generic(value: &GenericValue) -> Option<Self>;
}

impl GenericValue {
    pub fn as_typed<T: FromGenericValue>(&self) -> Option<T> {
        T::from_generic(self)
    }
}

macro_rules! impl_from_generic {
    ($t:ty, $variant:ident) => {
        impl FromGenericValue for $t {
            fn from_generic(value: &GenericValue) -> Option<Self> {
                match value {
                    GenericValue::$variant(v) => Some(v.clone()),
                    _ => None,
                }
            }
        }
    };
}

impl_from_generic!(bool, Bool);
impl_from_generic!(i64, Int64);
impl_from_generic!(f64, Float64);
impl_from_generic!(Complex64, Complex64);
// we do not implement Symbol extraction, since it is ambiguous - a symbol can be a Parameter or a ParameterVector

// Extracting tuples is a little more trick; we'll use macro for the easy case of Vec<T> for a specific T
impl<T: FromGenericValue> FromGenericValue for Vec<T> {
    fn from_generic(value: &GenericValue) -> Option<Self> {
        match value {
            GenericValue::Tuple(vec) => {
                let mut out = Vec::with_capacity(vec.len());
                for item in vec {
                    out.push(T::from_generic(item)?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

pub fn load_value(
    type_key: u8,
    bytes: &Bytes,
    qpy_data: &mut QPYReadData,
) -> PyResult<GenericValue> {
    match type_key {
        tags::BOOL => {
            let value: bool = bytes.try_into()?;
            Ok(GenericValue::Bool(value))
        }
        tags::INTEGER => {
            let value: i64 = bytes.try_into()?;
            Ok(GenericValue::Int64(value))
        }
        tags::FLOAT => {
            let value: f64 = bytes.try_into()?;
            Ok(GenericValue::Float64(value))
        }
        tags::COMPLEX => {
            let value: Complex64 = bytes.try_into()?;
            Ok(GenericValue::Complex64(value))
        }
        tags::PARAMETER => {
            let (parameter, _) = deserialize::<formats::ParameterPack>(bytes)?;
            let symbol = unpack_symbol(&parameter);
            Ok(GenericValue::ParameterExpressionSymbol(symbol))
        }
        tags::PARAMETER_VECTOR => {
            let (parameter_vector_element, _) = deserialize::<formats::ParameterVectorPack>(bytes)?;
            let symbol = unpack_parameter_vector(&parameter_vector_element, qpy_data)?;
            Ok(GenericValue::ParameterExpressionVectorSymbol(symbol))
        }
        tags::TUPLE => {
            let (elements_pack, _) = deserialize::<GenericDataSequencePack>(bytes)?;
            let values = unpack_generic_value_sequence(elements_pack, qpy_data)?;
            Ok(GenericValue::Tuple(values))
        }
        _ => Err(PyTypeError::new_err(format!(
            "py_dumps_value: Unhandled type_key: {}",
            type_key
        ))),
    }
}

/// serializes the generic value into bytes and also returns the identifying tag
pub fn serialize_generic_value(value: &GenericValue) -> PyResult<(u8, Bytes)> {
    Ok(match value {
        GenericValue::Bool(value) => (tags::BOOL, value.into()),
        GenericValue::Int64(value) => (tags::INTEGER, value.into()),
        GenericValue::Float64(value) => (tags::FLOAT, value.into()),
        GenericValue::Complex64(value) => (tags::COMPLEX, value.into()),
        GenericValue::ParameterExpressionSymbol(symbol) => {
            (tags::PARAMETER, serialize(&pack_symbol(symbol)))
        }
        GenericValue::ParameterExpressionVectorSymbol(symbol) => (
            tags::PARAMETER_VECTOR,
            serialize(&pack_parameter_vector(symbol)?),
        ),
        GenericValue::Tuple(values) => (
            tags::TUPLE,
            serialize(&pack_generic_value_sequence(values)?),
        ),
    })
}

// packing to GenericDataPack is somewhat wasteful in many cases, since given the type_key
// we usually know the byte length of the data and don't need to store it directly,
// but since that's the format currently in place in QPY we don't try to optimize
pub fn pack_generic_value(value: &GenericValue) -> PyResult<GenericDataPack> {
    let (type_key, data) = serialize_generic_value(value)?;
    Ok(GenericDataPack { type_key, data })
}

pub fn unpack_generic_value(
    value_pack: &GenericDataPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<GenericValue> {
    load_value(value_pack.type_key, &value_pack.data, qpy_data)
}

pub fn pack_generic_value_sequence(values: &[GenericValue]) -> PyResult<GenericDataSequencePack> {
    let elements = values
        .iter()
        .map(pack_generic_value)
        .collect::<PyResult<_>>()?;
    Ok(GenericDataSequencePack { elements })
}

pub fn unpack_generic_value_sequence(
    value_seqeunce_pack: GenericDataSequencePack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Vec<GenericValue>> {
    value_seqeunce_pack
        .elements
        .iter()
        .map(|data_pack| unpack_generic_value(data_pack, qpy_data))
        .collect()
}

pub mod circuit_instruction_types {
    pub const INSTRUCTION: u8 = b'i';
    pub const GATE: u8 = b'g';
    pub const PAULI_EVOL_GATE: u8 = b'p';
    pub const CONTROLLED_GATE: u8 = b'c';
    pub const ANNOTATED_OPERATION: u8 = b'a';
}

/// Each instruction type has a char representation in qpy
pub fn get_circuit_type_key(op: &PackedOperation) -> PyResult<u8> {
    match op.view() {
        OperationRef::StandardGate(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::StandardInstruction(_)
        | OperationRef::Instruction(_)
        | OperationRef::PauliProductMeasurement(_) => Ok(circuit_instruction_types::INSTRUCTION),
        OperationRef::Unitary(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::Gate(pygate) => Python::attach(|py| {
            let gate = pygate.gate.bind(py);
            if gate.is_instance(imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::PAULI_EVOL_GATE)
            } else if gate.is_instance(imports::CONTROLLED_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::CONTROLLED_GATE)
            } else {
                Ok(circuit_instruction_types::GATE)
            }
        }),
        OperationRef::Operation(operation) => Python::attach(|py| {
            if operation
                .operation
                .bind(py)
                .is_instance(imports::ANNOTATED_OPERATION.get_bound(py))?
            {
                Ok(circuit_instruction_types::ANNOTATED_OPERATION)
            } else {
                Err(PyErr::new::<PyValueError, _>(format!(
                    "Unable to determine circuit type key for {:?}",
                    operation
                )))
            }
        }),
    }
}

pub fn serialize_expression(exp: Expr, qpy_data: &QPYWriteData) -> PyResult<Bytes> {
    let packed_expression = formats::ExpressionPack {
        expression: exp,
        _phantom: Default::default(),
    };
    let serialized_expression = serialize_with_args(&packed_expression, (qpy_data,));
    Ok(serialized_expression)
}

pub fn deserialize_expression(raw_expression: &Bytes, qpy_data: &QPYReadData) -> PyResult<Expr> {
    let (exp_pack, _) = deserialize_with_args::<formats::ExpressionPack, (&QPYReadData,)>(
        raw_expression,
        (qpy_data,),
    )?;
    Ok(exp_pack.expression)
}

pub fn pack_standalone_var(
    var: &Var,
    usage: u8,
    version: u32,
    uuid_output: &mut u128,
) -> PyResult<formats::ExpressionVarDeclarationPack> {
    match var {
        Var::Standalone { uuid, name, ty } => {
            let exp_type = pack_expression_type(ty, version)?;
            *uuid_output = *uuid;
            let uuid_bytes = uuid.to_be_bytes();
            Ok(formats::ExpressionVarDeclarationPack {
                uuid_bytes,
                usage,
                exp_type,
                name: name.clone(),
            })
        }
        _ => Err(QpyError::new_err(format!(
            "attempted to pack as standalone var the non-standalone var {:?}",
            var
        ))),
    }
}

pub fn pack_stretch(stretch: &Stretch, usage: u8) -> formats::ExpressionVarDeclarationPack {
    formats::ExpressionVarDeclarationPack {
        uuid_bytes: stretch.uuid.to_be_bytes(),
        usage,
        exp_type: ExpressionType::Duration,
        name: stretch.name.clone(),
    }
}

// we convert the type to the serializable struct; this amounts to simple copy unless
// there's a field not supported in the expected version
fn pack_expression_type(exp_type: &Type, version: u32) -> PyResult<ExpressionType> {
    match exp_type {
        Type::Bool => Ok(ExpressionType::Bool),
        Type::Duration => {
            if version >= 14 {
                Ok(ExpressionType::Duration)
            } else {
                Err(UnsupportedFeatureForVersion::new_err((
                    "duration-typed expressions",
                    14,
                    version,
                )))
            }
        }
        Type::Float => {
            if version >= 14 {
                Ok(ExpressionType::Float)
            } else {
                Err(UnsupportedFeatureForVersion::new_err((
                    "float-typed expressions",
                    14,
                    version,
                )))
            }
        }
        Type::Uint(width) => Ok(ExpressionType::Uint(*width as u32)),
    }
}

pub struct DumpedPyValue {
    pub data_type: u8,
    pub data: Bytes,
}

impl DumpedPyValue {
    pub fn to_python(&self, py: Python<'_>, qpy_data: &mut QPYReadData) -> PyResult<Py<PyAny>> {
        Ok(match self.data_type {
            tags::INTEGER => {
                let value: i64 = (&self.data).try_into()?;
                PyInt::new(py, value).into()
            }
            tags::FLOAT => PyFloat::new(py, (&self.data).try_into()?).into(),
            tags::COMPLEX => {
                let (real, imag) = (&self.data).try_into()?;
                PyComplex::from_doubles(py, real, imag).into()
            }
            tags::RANGE => py_deserialize_range(py, &self.data)?,
            tags::TUPLE => py_unpack_generic_sequence_to_tuple(
                py,
                deserialize::<formats::GenericDataSequencePack>(&self.data)?.0,
                qpy_data,
            )?,
            tags::NUMPY_OBJ => {
                let np = py.import("numpy")?;
                let io = py.import("io")?;
                let buffer = io.call_method0("BytesIO")?;
                buffer.call_method1("write", (self.data.clone(),))?;
                buffer.call_method1("seek", (0,))?;
                np.call_method1("load", (buffer,))?.unbind()
            }
            tags::MODIFIER => {
                let (packed_modifier, _) = deserialize::<formats::ModifierPack>(&self.data)?;
                py_unpack_modifier(py, &packed_modifier)?
            }
            tags::STRING => {
                let data_string: &str = (&self.data).try_into()?;
                data_string.into_py_any(py)?
            }
            tags::EXPRESSION => deserialize_expression(&self.data, qpy_data)?.into_py_any(py)?,
            tags::NULL => py.None(),
            tags::CASE_DEFAULT => imports::CASE_DEFAULT.get(py).clone(),
            tags::CIRCUIT => unpack_circuit(
                py,
                &deserialize::<formats::QPYFormatV15>(&self.data)?.0,
                qpy_data.version,
                py.None().bind(py),
                qpy_data.use_symengine,
                qpy_data.annotation_handler.annotation_factories,
            )?
            .unbind(),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Dumped Value to python: Unhandled type_key: {}",
                    self.data_type
                )));
            }
        })
    }
}

impl fmt::Debug for DumpedPyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DumpedPyValue")
            .field("type", &self.data_type)
            .field("data", &self.data.to_hex_string())
            .finish()
    }
}

pub fn bytes_to_py_uuid(py: Python, bytes: [u8; 16]) -> PyResult<Py<PyAny>> {
    let uuid_module = py.import("uuid")?;
    let py_bytes = PyBytes::new(py, &bytes);
    let kwargs = PyDict::new(py);
    kwargs.set_item("bytes", py_bytes)?;
    Ok(uuid_module.call_method("UUID", (), Some(&kwargs))?.unbind())
}
