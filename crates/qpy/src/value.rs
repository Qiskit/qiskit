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
use binrw::{binrw, BinRead, BinResult, BinWrite, Endian};
use hashbrown::HashMap;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyAny, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyo3::IntoPyObjectExt;
use pyo3::{intern, prelude::*};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::classical::expr::Expr;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{OperationRef, Param};
use qiskit_circuit::packed_instruction::PackedOperation;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::circuits::{deserialize_circuit, pack_circuit};
use crate::formats;
use crate::params::{
    pack_parameter, pack_parameter_expression, pack_parameter_vector, unpack_parameter,
    unpack_parameter_expression, unpack_parameter_vector,
};
use crate::{QpyError, UnsupportedFeatureForVersion};
use core::fmt;
use std::fmt::Debug;
use std::io::{Cursor, Read, Seek, Write};
use uuid::Uuid;

const QPY_VERSION: u32 = 15;

pub struct QPYWriteData {
    pub version: u32,
    pub _use_symengine: bool,
    pub clbit_indices: Py<PyDict>,
    pub standalone_var_indices: Py<PyDict>,
    pub annotation_handler: AnnotationHandler,
}

pub struct QPYReadData<'a> {
    pub circuit_data: &'a mut CircuitData,
    pub version: u32,
    pub use_symengine: bool,
    pub clbit_indices: Py<PyDict>,
    pub cregs: Py<PyDict>,
    pub standalone_vars: Py<PyList>,
    pub vectors: HashMap<Uuid, (Py<PyAny>, Vec<u64>)>,
    pub annotation_handler: AnnotationHandler,
}

pub mod tags {
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

pub fn serialize<T>(value: &T) -> PyResult<Bytes>
where
    T: BinWrite + WriteEndian + Debug,
    for<'a> <T as BinWrite>::Args<'a>: Default,
{
    let mut buffer = Cursor::new(Vec::new());
    value.write(&mut buffer).unwrap();
    let result: Bytes = buffer.into();
    Ok(result)
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
    A: Clone + Debug,
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

pub fn get_type_key(py_object: &Bound<PyAny>) -> PyResult<u8> {
    let py = py_object.py();
    if py_object
        .is_instance(imports::PARAMETER_VECTOR_ELEMENT.get_bound(py))
        .unwrap()
    {
        return Ok(tags::PARAMETER_VECTOR);
    } else if py_object
        .is_instance(imports::PARAMETER.get_bound(py))
        .unwrap()
    {
        return Ok(tags::PARAMETER);
    } else if py_object.is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))? {
        return Ok(tags::PARAMETER_EXPRESSION);
    } else if py_object.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
        return Ok(tags::CIRCUIT);
    } else if py_object.is_instance(imports::CLBIT.get_bound(py))?
        || py_object.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))?
    {
        return Ok(tags::REGISTER);
    } else if py_object.extract::<Expr>().is_ok() {
        return Ok(tags::EXPRESSION);
    } else if py_object.is_instance(imports::BUILTIN_RANGE.get_bound(py))? {
        return Ok(tags::RANGE);
    } else if py_object.is_instance(imports::NUMPY_ARRAY.get_bound(py))? {
        return Ok(tags::NUMPY_OBJ);
    } else if py_object.is_instance(imports::MODIFIER.get_bound(py))? {
        return Ok(tags::MODIFIER);
    } else if py_object.is_instance_of::<PyInt>() {
        return Ok(tags::INTEGER);
    } else if py_object.is_instance_of::<PyFloat>() {
        return Ok(tags::FLOAT);
    } else if py_object.is_instance_of::<PyComplex>() {
        return Ok(tags::COMPLEX);
    } else if py_object.is_instance_of::<PyString>() {
        return Ok(tags::STRING);
    } else if py_object.is_instance_of::<PyTuple>() {
        return Ok(tags::TUPLE);
    } else if py_object.is(imports::CASE_DEFAULT.get_bound(py)) {
        return Ok(tags::CASE_DEFAULT);
    } else if py_object.is_none() {
        return Ok(tags::NULL);
    }

    Err(PyTypeError::new_err(format!(
        "Unidentified type_key for: {}",
        py_object
    )))
}

pub fn dumps_value(py_object: &Bound<PyAny>, qpy_data: &QPYWriteData) -> PyResult<(u8, Bytes)> {
    let py = py_object.py();
    let type_key: u8 = get_type_key(py_object)?;
    let value: Bytes = match type_key {
        tags::INTEGER => py_object.extract::<i64>()?.to_be_bytes().into(),
        tags::FLOAT => py_object.extract::<f64>()?.to_be_bytes().into(),
        tags::COMPLEX => {
            let complex_num = py_object.downcast::<PyComplex>()?;
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&complex_num.real().to_be_bytes());
            bytes.extend_from_slice(&complex_num.imag().to_be_bytes());
            bytes.into()
        }
        tags::RANGE => serialize_range(py_object)?,
        tags::TUPLE => serialize(&pack_generic_sequence(py_object, qpy_data)?)?,
        tags::PARAMETER => serialize(&pack_parameter(py_object)?)?,
        tags::PARAMETER_VECTOR => serialize(&pack_parameter_vector(py_object)?)?,
        tags::PARAMETER_EXPRESSION => serialize(&pack_parameter_expression(py_object, qpy_data)?)?,
        tags::NUMPY_OBJ => {
            let np = py.import("numpy")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;
            np.call_method1("save", (&buffer, py_object))?;
            buffer.call_method0("getvalue")?.extract::<Bytes>()?
        }
        tags::MODIFIER => serialize(&pack_modifier(py_object)?)?,
        tags::STRING => py_object.extract::<String>()?.into(),
        tags::EXPRESSION => serialize_expression(py_object, qpy_data)?,
        tags::NULL | tags::CASE_DEFAULT => Bytes::new(),
        tags::CIRCUIT => serialize(&pack_circuit(
            py_object,
            py.None().bind(py),
            false,
            QPY_VERSION,
            qpy_data.annotation_handler.annotation_factories.clone(),
        )?)?,
        _ => {
            return Err(PyTypeError::new_err(format!(
                "dumps_value: Unhandled type_key: {}",
                type_key
            )))
        }
    };
    Ok((type_key, value))
}

pub fn pack_generic_data(
    py_data: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::GenericDataPack> {
    let (type_key, data) = dumps_value(py_data, qpy_data)?;
    Ok(formats::GenericDataPack { type_key, data })
}

pub fn unpack_generic_data(
    py: Python,
    data_pack: &formats::GenericDataPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Py<PyAny>> {
    DumpedValue {
        data_type: data_pack.type_key,
        data: data_pack.data.clone(),
    }
    .to_python(py, qpy_data)
}

pub fn pack_generic_sequence(
    py_sequence: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::GenericDataSequencePack> {
    let elements: Vec<formats::GenericDataPack> = py_sequence
        .try_iter()?
        .map(|possible_data_item| {
            let data_item = possible_data_item?;
            pack_generic_data(&data_item, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    Ok(formats::GenericDataSequencePack { elements })
}

pub fn unpack_generic_sequence_to_tuple(
    py: Python,
    packed_sequence: formats::GenericDataSequencePack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Py<PyAny>> {
    let elements: Vec<Py<PyAny>> = packed_sequence
        .elements
        .iter()
        .map(|data_pack| unpack_generic_data(py, data_pack, qpy_data))
        .collect::<PyResult<_>>()?;
    elements.into_py_any(py)
}

pub mod circuit_instruction_types {
    pub const INSTRUCTION: u8 = b'i';
    pub const GATE: u8 = b'g';
    pub const PAULI_EVOL_GATE: u8 = b'p';
    pub const CONTROLLED_GATE: u8 = b'c';
    pub const ANNOTATED_OPERATION: u8 = b'a';
}

pub fn get_circuit_type_key(py: Python, op: &PackedOperation) -> PyResult<u8> {
    match op.view() {
        OperationRef::StandardGate(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::StandardInstruction(_) | OperationRef::Instruction(_) => {
            Ok(circuit_instruction_types::INSTRUCTION)
        }
        OperationRef::Unitary(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::Gate(pygate) => {
            let gate = pygate.gate.bind(py);
            if gate.is_instance(imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::PAULI_EVOL_GATE)
            } else if gate.is_instance(imports::CONTROLLED_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::CONTROLLED_GATE)
            } else {
                Ok(circuit_instruction_types::GATE)
            }
        }
        OperationRef::Operation(operation) => {
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
        }
    }
}

fn serialize_range(py_object: &Bound<PyAny>) -> PyResult<Bytes> {
    let start = py_object.getattr("start")?.extract::<i64>()?;
    let stop = py_object.getattr("stop")?.extract::<i64>()?;
    let step = py_object.getattr("step")?.extract::<i64>()?;
    let range_pack = formats::RangePack { start, stop, step };
    let mut buffer = Cursor::new(Vec::new());
    range_pack.write(&mut buffer).unwrap();
    Ok(buffer.into())
}

fn deserialize_range(py: Python, raw_range: &Bytes) -> PyResult<Py<PyAny>> {
    let range_pack = deserialize::<formats::RangePack>(raw_range)?.0;
    Ok(imports::BUILTIN_RANGE
        .get_bound(py)
        .call1((range_pack.start, range_pack.stop, range_pack.step))?
        .unbind())
}

fn serialize_expression(py_object: &Bound<PyAny>, qpy_data: &QPYWriteData) -> PyResult<Bytes> {
    let py = py_object.py();
    let clbit_indices = PyDict::new(py);
    for (key, val) in qpy_data.clbit_indices.bind(py).iter() {
        let index = val.getattr("index")?;
        clbit_indices.set_item(key, index)?;
    }
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    let value = py.import("qiskit.qpy.binary_io.value")?;
    value.call_method1(
        "_write_expr",
        (
            &buffer,
            py_object,
            clbit_indices,
            &qpy_data.standalone_var_indices,
            qpy_data.version,
        ),
    )?;
    let result = buffer.call_method0("getvalue")?.extract::<Bytes>()?;
    Ok(result)
}

fn deserialize_expression(
    py: Python,
    raw_expression: &Bytes,
    qpy_data: &QPYReadData,
) -> PyResult<Py<PyAny>> {
    let clbit_indices = PyDict::new(py);
    for (key, val) in qpy_data.clbit_indices.bind(py).iter() {
        let index = val.getattr("index")?;
        clbit_indices.set_item(index, key)?;
    }
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    buffer.call_method1("write", (raw_expression.clone(),))?;
    buffer.call_method1("seek", (0,))?;
    let value = py.import("qiskit.qpy.binary_io.value")?;
    let expression = value.call_method1(
        "_read_expr",
        (
            &buffer,
            clbit_indices,
            &qpy_data.cregs,
            &qpy_data.standalone_vars,
        ),
    )?;
    Ok(expression.unbind())
}

fn pack_modifier(modifier: &Bound<PyAny>) -> PyResult<formats::ModifierPack> {
    let py = modifier.py();
    let module = py.import("qiskit.circuit.annotated_operation")?;
    if modifier.is_instance(&module.getattr("InverseModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: modifier_types::INVERSE,
            num_ctrl_qubits: 0,
            ctrl_state: 0,
            power: 0.0,
        })
    } else if modifier.is_instance(&module.getattr("ControlModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: modifier_types::CONTROL,
            num_ctrl_qubits: modifier.getattr("num_ctrl_qubits")?.extract::<u32>()?,
            ctrl_state: modifier.getattr("ctrl_state")?.extract::<u32>()?,
            power: 0.0,
        })
    } else if modifier.is_instance(&module.getattr("PowerModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: modifier_types::POWER,
            num_ctrl_qubits: 0,
            ctrl_state: 0,
            power: modifier.getattr("power")?.extract::<f64>()?,
        })
    } else {
        Err(PyTypeError::new_err("Unsupported modifier."))
    }
}

fn unpack_modifier(py: Python, packed_modifier: &formats::ModifierPack) -> PyResult<Py<PyAny>> {
    match packed_modifier.modifier_type {
        modifier_types::INVERSE => Ok(imports::INVERSE_MODIFIER.get_bound(py).call0()?.unbind()),
        modifier_types::CONTROL => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                intern!(py, "num_ctrl_qubits"),
                packed_modifier.num_ctrl_qubits,
            )?;
            kwargs.set_item(intern!(py, "ctrl_state"), packed_modifier.ctrl_state)?;
            Ok(imports::CONTROL_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
        modifier_types::POWER => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "power"), packed_modifier.power)?;
            Ok(imports::POWER_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
        _ => Err(PyTypeError::new_err("Unsupported modifier.")),
    }
}

pub fn pack_standalone_var(
    var: &Bound<PyAny>,
    usage: u8,
    version: u32,
) -> PyResult<formats::ExpressionVarDeclarationPack> {
    let name = var.getattr("name")?.extract::<String>()?;
    let exp_type = pack_expression_type(&var.getattr("type")?, version)?;
    let uuid_bytes = var
        .getattr("var")?
        .getattr("bytes")?
        .extract::<[u8; 16]>()?;
    Ok(formats::ExpressionVarDeclarationPack {
        uuid_bytes,
        usage,
        exp_type,
        name,
    })
}

fn pack_expression_type(exp_type: &Bound<PyAny>, version: u32) -> PyResult<ExpressionType> {
    let kind = exp_type.getattr("kind")?;
    let py = exp_type.py();
    let types_module = py.import("qiskit.circuit.classical.types")?;
    if kind.is(&types_module.getattr("Bool")?) {
        Ok(ExpressionType::Bool)
    } else if kind.is(&types_module.getattr("Uint")?) {
        let width = exp_type.getattr("width")?.extract::<u32>()?;
        Ok(ExpressionType::Uint(width))
    } else if kind.is(&types_module.getattr("Float")?) {
        if version < 14 {
            return Err(UnsupportedFeatureForVersion::new_err((
                "float-typed expressions",
                14,
                version,
            )));
        }
        Ok(ExpressionType::Float)
    } else if kind.is(&types_module.getattr("Duration")?) {
        if version < 14 {
            return Err(UnsupportedFeatureForVersion::new_err((
                "duration-typed expressions",
                14,
                version,
            )));
        }
        Ok(ExpressionType::Duration)
    } else {
        return Err(QpyError::new_err((format!(
            "unhandled Type object {:?}",
            exp_type
        ),)));
    }
}
pub struct DumpedValue {
    pub data_type: u8,
    pub data: Bytes,
}

impl DumpedValue {
    pub fn from(py_object: &Bound<PyAny>, qpy_data: &QPYWriteData) -> PyResult<Self> {
        let (data_type, data) = dumps_value(py_object, qpy_data)?;
        Ok(DumpedValue { data_type, data })
    }
    pub fn write<W: Write>(
        value: &DumpedValue,
        writer: &mut W,
        _endian: Endian,
        _args: (),
    ) -> binrw::BinResult<()> {
        Ok(writer.write_all(&value.data)?)
    }

    pub fn read<R: Read + Seek>(
        reader: &mut R,
        _endian: Endian,
        (len, data_type): (usize, u8),
    ) -> BinResult<DumpedValue> {
        let mut buf = Bytes(vec![0u8; len]);
        reader.read_exact(&mut buf)?;
        Ok(DumpedValue {
            data_type,
            data: buf,
        })
    }

    pub fn to_python(&self, py: Python<'_>, qpy_data: &mut QPYReadData) -> PyResult<PyObject> {
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

            // }
            tags::RANGE => deserialize_range(py, &self.data)?,
            tags::TUPLE => unpack_generic_sequence_to_tuple(
                py,
                deserialize::<formats::GenericDataSequencePack>(&self.data)?.0,
                qpy_data,
            )?,
            tags::PARAMETER => {
                let (parameter, _) = deserialize::<formats::ParameterPack>(&self.data)?;
                unpack_parameter(py, &parameter)?
            }
            tags::PARAMETER_VECTOR => {
                let (parameter_vector, _) =
                    deserialize::<formats::ParameterVectorPack>(&self.data)?;
                unpack_parameter_vector(py, &parameter_vector, qpy_data)?
            }

            tags::PARAMETER_EXPRESSION => {
                let (parameter_expression, _) =
                    deserialize::<formats::ParameterExpressionPack>(&self.data)?;
                unpack_parameter_expression(py, parameter_expression, qpy_data)?
            }
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
                unpack_modifier(py, &packed_modifier)?
            }
            tags::STRING => {
                let data_string: &str = (&self.data).try_into()?;
                data_string.into_py_any(py)?
            }
            tags::EXPRESSION => deserialize_expression(py, &self.data, qpy_data)?,
            tags::NULL => py.None(),
            tags::CASE_DEFAULT => imports::CASE_DEFAULT.get(py).clone(),
            tags::CIRCUIT => deserialize_circuit(
                py,
                &self.data,
                qpy_data.version,
                py.None().bind(py),
                qpy_data.use_symengine,
                qpy_data.annotation_handler.annotation_factories.clone(),
            )?
            .0
            .unbind(),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Dumped Value to python: Unhandled type_key: {}",
                    self.data_type
                )))
            }
        })
    }
    pub fn to_param(&self, py: Python, qpy_data: &mut QPYReadData) -> PyResult<Param> {
        match self.data_type {
            tags::FLOAT => Ok(Param::Float((&self.data).try_into()?)),
            tags::PARAMETER_EXPRESSION | tags::PARAMETER | tags::PARAMETER_VECTOR => {
                Ok(Param::ParameterExpression(self.to_python(py, qpy_data)?))
            }
            _ => Ok(Param::Obj(self.to_python(py, qpy_data)?)),
        }
    }
}

impl fmt::Debug for DumpedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DumpedValue")
            .field("type", &self.data_type)
            .field("data", &self.data.to_hex_string())
            .finish()
    }
}

pub fn bytes_to_uuid(py: Python, bytes: [u8; 16]) -> PyResult<PyObject> {
    let uuid_module = py.import("uuid")?;
    let py_bytes = PyBytes::new(py, &bytes);
    let kwargs = PyDict::new(py);
    kwargs.set_item("bytes", py_bytes)?;
    Ok(uuid_module.call_method("UUID", (), Some(&kwargs))?.unbind())
}
