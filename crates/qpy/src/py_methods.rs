// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Methods for QPY serialization working directly with Python-based data
use binrw::Endian;
use numpy::Complex64;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyComplex, PyDict, PyFloat, PyInt, PyIterator, PyList, PyString, PyTuple,
};
use qiskit_circuit::classical::expr::Expr;
use std::num::NonZero;
use std::sync::Arc;

use qiskit_circuit::bit::{ClassicalRegister, ShareableClbit};
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Operation, OperationRef, PyRange};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter::parameter_expression::{
    PyParameter, PyParameterExpression, PyParameterVectorElement,
};
use qiskit_quantum_info::sparse_observable::PySparseObservable;
use uuid::Uuid;

use crate::bytes::Bytes;
use crate::circuit_writer::standard_instruction_class_name;
use crate::formats;
use crate::value::{
    GenericValue, ModifierType, ParamRegisterValue, QPYWriteData, ValueType,
    serialize_generic_value,
};

pub const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";
pub const PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME: &str = "PauliProductMeasurement";

fn is_python_gate(py: Python, op: &PackedOperation, python_gate: &Bound<PyAny>) -> PyResult<bool> {
    match op.view() {
        OperationRef::Gate(pygate) => {
            if pygate.instruction.bind(py).is_instance(python_gate)? {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

/// custom gates have unique UUID attached to their name
/// this method recognizes whether we have such a gate and returns a unique name for it
/// since custom gates are implemented in python, this is a heavy python-space function
pub(crate) fn recognize_custom_operation(
    op: &PackedOperation,
    name: &String,
) -> PyResult<Option<String>> {
    Python::attach(|py| {
        let library = py.import("qiskit.circuit.library")?;
        let circuit_mod = py.import("qiskit.circuit")?;
        let controlflow = py.import("qiskit.circuit.controlflow")?;

        if (!library.hasattr(name)?
            && !circuit_mod.hasattr(name)?
            && !controlflow.hasattr(name)?
            && (name != "Clifford" && name != PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME))
            || name == "Gate"
            || name == "Instruction"
            || is_python_gate(py, op, imports::BLUEPRINT_CIRCUIT.get_bound(py))?
        {
            // Assign a uuid to each instance of a custom operation
            let new_name = if !["ucrx_dg", "ucry_dg", "ucrz_dg"].contains(&op.name()) {
                format!("{}_{}", &op.name(), Uuid::new_v4().as_simple())
            } else {
                // ucr*_dg gates can have different numbers of parameters,
                // the uuid is appended to avoid storing a single definition
                // in circuits with multiple ucr*_dg gates. For legacy reasons
                // the uuid is stored in a different format as this was done
                // prior to QPY 11.
                format!("{}_{}", &op.name(), Uuid::new_v4())
            };
            return Ok(Some(new_name));
        }

        if ["ControlledGate", "AnnotatedOperation"].contains(&name.as_str())
            || is_python_gate(py, op, imports::MCMT_GATE.get_bound(py))?
        {
            return Ok(Some(format!("{}_{}", op.name(), Uuid::new_v4())));
        }

        if is_python_gate(py, op, imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
            return Ok(Some(format!("###PauliEvolutionGate_{}", Uuid::new_v4())));
        }

        Ok(None)
    })
}

/// when trying to instantiate nonstandard gates, we turn to the relevant python clas
/// this function obtains the class based on the gate class name
pub(crate) fn get_python_gate_class<'a>(
    py: Python<'a>,
    gate_class_name: &String,
) -> PyResult<Bound<'a, PyAny>> {
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let control_flow = py.import("qiskit.circuit.controlflow")?;
    if library.hasattr(gate_class_name)? {
        library.getattr(gate_class_name)
    } else if circuit_mod.hasattr(gate_class_name)? {
        circuit_mod.getattr(gate_class_name)
    } else if control_flow.hasattr(gate_class_name)? {
        control_flow.getattr(gate_class_name)
    } else if gate_class_name == "Clifford" {
        Ok(imports::CLIFFORD.get_bound(py).clone())
    } else if gate_class_name == "pauli_product_measurement" {
        Ok(imports::PAULI_PRODUCT_MEASUREMENT.get_bound(py).clone())
    } else {
        Err(PyIOError::new_err(format!(
            "Gate class not found: {:?}",
            gate_class_name
        )))
    }
}

// serializes python metadata to JSON using a python JSON serializer
pub(crate) fn serialize_metadata(
    metadata_opt: &Option<Bound<PyAny>>,
    metadata_serializer: Option<&Bound<PyAny>>,
) -> PyResult<Bytes> {
    match metadata_opt {
        None => Ok(Bytes::new()),
        Some(metadata) => {
            let py = metadata.py();
            let none = py.None();
            let py_serializer = metadata_serializer.unwrap_or(none.bind(py));
            let json = py.import("json")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
            kwargs.set_item("cls", py_serializer)?;
            Ok(json
                .call_method("dumps", (metadata,), Some(&kwargs))?
                .extract::<String>()?
                .into())
        }
    }
}

// helper method to extract attribute from a py_object
pub(crate) fn getattr_or_none<'py>(
    py_object: &'py Bound<'py, PyAny>,
    name: &str,
) -> Option<Bound<'py, PyAny>> {
    match py_object.getattr(name) {
        Ok(val) => {
            if val.is_none() {
                None
            } else {
                Some(val)
            }
        }
        Err(_) => None,
    }
}

pub(crate) fn py_serialize_numpy_object(py_object: &Py<PyAny>) -> PyResult<Bytes> {
    Python::attach(|py| {
        let np = py.import("numpy")?;
        let io = py.import("io")?;
        let buffer = io.call_method0("BytesIO")?;
        np.call_method1("save", (&buffer, py_object))?;
        buffer.call_method0("getvalue")?.extract::<Bytes>()
    })
}

pub(crate) fn py_deserialize_numpy_object(data: &Bytes) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let np = py.import("numpy")?;
        let io = py.import("io")?;
        let buffer = io.call_method0("BytesIO")?;
        buffer.call_method1("write", (data.clone(),))?;
        buffer.call_method1("seek", (0,))?;
        Ok(np.call_method1("load", (buffer,))?.unbind())
    })
}

fn pack_sparse_pauli_op(
    operator: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::PauliDataPack> {
    if operator.is_instance_of::<PySparseObservable>() {
        let py_sparse_observable: PyRef<PySparseObservable> = operator.extract()?;
        let sparse_observable = py_sparse_observable
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Can't extract sparse observable data"))?;
        let num_qubits = sparse_observable.num_qubits();
        let coeff_data = sparse_observable
            .coeffs()
            .iter()
            .flat_map(|coeff| [coeff.re, coeff.im])
            .collect();
        let bitterm_data = sparse_observable
            .bit_terms()
            .iter()
            .map(|&bitterm| bitterm as u16)
            .collect();
        let inds_data = sparse_observable.indices().to_vec();
        let bounds_data = sparse_observable
            .boundaries()
            .iter()
            .map(|&boundary| boundary as u64)
            .collect();
        let sparse_observable_pack = formats::SparsePauliObservableElemPack {
            num_qubits,
            coeff_data,
            bitterm_data,
            inds_data,
            bounds_data,
        };
        Ok(formats::PauliDataPack::V17(
            formats::PauliDataPackV17::SparseObservable(sparse_observable_pack),
        ))
    } else {
        // this is the case of SparsePauliOp, which we convert to a numpy list
        let op_as_np_list = operator.call_method1("to_list", (true,))?;
        let value = py_convert_to_generic_value(&op_as_np_list)?;
        let (_, data) = serialize_generic_value(&value, qpy_data)?;
        let pack = formats::SparsePauliOpListElemPack { data };
        Ok(formats::PauliDataPack::V17(
            formats::PauliDataPackV17::SparsePauliOp(pack),
        ))
    }
}

pub(crate) fn py_pack_pauli_evolution_gate(
    evolution_gate: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::PauliEvolutionDefPack> {
    let py = evolution_gate.py();
    let operators = evolution_gate.getattr("operator")?;
    let mut standalone = false;
    let operator_list: Bound<PyList> = if !operators.is_instance_of::<PyList>() {
        standalone = true;
        PyList::new(py, [operators])?
    } else {
        operators.cast()?.clone()
    };
    let pauli_data = operator_list
        .iter()
        .map(|operator| pack_sparse_pauli_op(&operator, qpy_data))
        .collect::<PyResult<_>>()?;

    let time_value = py_convert_to_generic_value(&evolution_gate.getattr("time")?)?;
    let (time_type, time_data) = serialize_generic_value(&time_value, qpy_data)?;
    let synth_class = evolution_gate
        .getattr("synthesis")?
        .get_type()
        .getattr("__name__")?;
    let settings_dict = evolution_gate.getattr("synthesis")?.getattr("settings")?;
    let json = py.import("json")?;
    let args = PyDict::new(py);
    args.set_item("class", synth_class)?;
    args.set_item("settings", settings_dict)?;
    let synth_data: Bytes = json
        .call_method1("dumps", (args,))?
        .extract::<String>()?
        .into();

    let standalone_op = standalone as u8;
    Ok(formats::PauliEvolutionDefPack {
        standalone_op,
        time_type,
        pauli_data,
        time_data,
        synth_data,
    })
}

pub(crate) fn gate_class_name(op: &PackedOperation) -> PyResult<String> {
    Python::attach(|py| {
        let name = match op.view() {
            // getting __name__ for standard gates and instructions should
            // eventually be replaced with a Rust-side mapping
            OperationRef::StandardGate(gate) => Ok(imports::get_std_gate_class_name(&gate)),
            OperationRef::StandardInstruction(inst) => {
                Ok(standard_instruction_class_name(&inst).to_string())
            }
            OperationRef::Gate(pygate) => pygate
                .instruction
                .bind(py)
                .getattr(intern!(py, "__class__"))?
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
            OperationRef::Instruction(pyinst) => pyinst
                .instruction
                .bind(py)
                .getattr(intern!(py, "__class__"))?
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
            OperationRef::Unitary(_) => Ok(UNITARY_GATE_CLASS_NAME.to_string()),
            OperationRef::Operation(py_op) => py_op
                .instruction
                .bind(py)
                .getattr(intern!(py, "__class__"))?
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
            OperationRef::PauliProductMeasurement(_) => {
                Ok(String::from(PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME))
            }
            OperationRef::ControlFlow(inst) => Ok(inst.name().to_string()),
        }?;
        Ok(name)
    })
}

pub(crate) fn py_get_type_key(py_object: &Bound<PyAny>) -> PyResult<ValueType> {
    let py: Python<'_> = py_object.py();
    if py_object
        .is_instance(imports::PARAMETER_VECTOR_ELEMENT.get_bound(py))
        .unwrap()
    {
        return Ok(ValueType::ParameterVector);
    } else if py_object
        .is_instance(imports::PARAMETER.get_bound(py))
        .unwrap()
    {
        return Ok(ValueType::Parameter);
    } else if py_object.is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))? {
        return Ok(ValueType::ParameterExpression);
    } else if py_object.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
        return Ok(ValueType::Circuit);
    } else if py_object.is_instance(imports::CLBIT.get_bound(py))?
        || py_object.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))?
    {
        return Ok(ValueType::Register);
    } else if py_object.extract::<classical::expr::Expr>().is_ok() {
        return Ok(ValueType::Expression);
    } else if py_object.is_instance(imports::BUILTIN_RANGE.get_bound(py))? {
        return Ok(ValueType::Range);
    } else if py_object.is_instance(imports::NUMPY_ARRAY.get_bound(py))? {
        return Ok(ValueType::NumpyObject);
    } else if py_object.is_instance(imports::MODIFIER.get_bound(py))? {
        return Ok(ValueType::Modifier);
    } else if py_object.is_instance_of::<PyInt>() {
        return Ok(ValueType::Integer);
    } else if py_object.is_instance_of::<PyFloat>() {
        return Ok(ValueType::Float);
    } else if py_object.is_instance_of::<PyComplex>() {
        return Ok(ValueType::Complex);
    } else if py_object.is_instance_of::<PyString>() {
        return Ok(ValueType::String);
    } else if py_object.is_instance_of::<PyTuple>() || py_object.is_instance_of::<PyList>() {
        return Ok(ValueType::Tuple);
    } else if py_object.is(imports::CASE_DEFAULT.get_bound(py)) {
        return Ok(ValueType::CaseDefault);
    } else if py_object.is_none() {
        return Ok(ValueType::Null);
    }

    Err(PyTypeError::new_err(format!(
        "Unidentified type_key for: {}",
        py_object
    )))
}

pub(crate) fn py_convert_to_generic_value(py_object: &Bound<PyAny>) -> PyResult<GenericValue> {
    let type_key = py_get_type_key(py_object)?;
    match type_key {
        ValueType::Bool => Ok(GenericValue::Bool(py_object.extract::<bool>()?)),
        ValueType::Integer => Ok(GenericValue::Int64(py_object.extract::<i64>()?)),
        ValueType::Float => Ok(GenericValue::Float64(py_object.extract::<f64>()?)),
        ValueType::Complex => Ok(GenericValue::Complex64(py_object.extract::<Complex64>()?)),
        ValueType::String => Ok(GenericValue::String(py_object.extract::<String>()?)),
        ValueType::Expression => Ok(GenericValue::Expression(py_object.extract::<Expr>()?)),
        ValueType::CaseDefault => Ok(GenericValue::CaseDefault),
        ValueType::Null => Ok(GenericValue::Null),
        ValueType::Parameter => Ok(GenericValue::ParameterExpressionSymbol(
            py_object.extract::<PyParameter>()?.symbol().clone(),
        )),
        ValueType::ParameterVector => Ok(GenericValue::ParameterExpressionVectorSymbol(
            py_object
                .extract::<PyParameterVectorElement>()?
                .symbol()
                .clone(),
        )),
        ValueType::ParameterExpression => Ok(GenericValue::ParameterExpression(Arc::new(
            py_object.extract::<PyParameterExpression>()?.inner,
        ))),
        ValueType::Circuit => Ok(GenericValue::Circuit(py_object.clone().unbind())),
        ValueType::Tuple => {
            let elements: Vec<GenericValue> = py_object
                .try_iter()?
                .map(|data_item| {
                    // let data_item = possible_data_item?;
                    py_convert_to_generic_value(&data_item?)
                })
                .collect::<PyResult<_>>()?;
            Ok(GenericValue::Tuple(elements))
        }
        ValueType::Range => {
            let start = py_object.getattr("start")?.extract::<isize>()?;
            let stop = py_object.getattr("stop")?.extract::<isize>()?;
            let step = py_object.getattr("step")?.extract::<isize>()?;
            let step = NonZero::new(step).expect("Python does not allow zero steps");
            let range = PyRange { start, stop, step };
            Ok(GenericValue::Range(range))
        }
        // the python-managed data types
        ValueType::NumpyObject => Ok(GenericValue::NumpyObject(py_object.clone().unbind())),
        ValueType::Modifier => Ok(GenericValue::Modifier(py_object.clone().unbind())),
        ValueType::Register => {
            if let Ok(clbit) = py_object.extract::<ShareableClbit>() {
                Ok(GenericValue::Register(ParamRegisterValue::ShareableClbit(
                    clbit,
                )))
            } else if let Ok(reg) = py_object.extract::<ClassicalRegister>() {
                Ok(GenericValue::Register(ParamRegisterValue::Register(reg)))
            } else {
                Err(PyValueError::new_err("Could not read python register"))
            }
        }
    }
}

pub(crate) fn py_convert_from_generic_value(value: &GenericValue) -> PyResult<Py<PyAny>> {
    Python::attach(|py| match value {
        GenericValue::Bool(value) => value.into_py_any(py),
        GenericValue::Int64(value) => value.into_py_any(py),
        GenericValue::Float64(value) => value.into_py_any(py),
        GenericValue::Complex64(value) => value.into_py_any(py),
        GenericValue::String(value) => value.into_py_any(py),
        GenericValue::Expression(exp) => exp.clone().into_py_any(py),
        GenericValue::CaseDefault => Ok(imports::CASE_DEFAULT.get(py).clone()),
        GenericValue::Null => Ok(py.None()),
        GenericValue::ParameterExpressionSymbol(symbol) => symbol.clone().into_py_any(py),
        GenericValue::ParameterExpressionVectorSymbol(symbol) => symbol.clone().into_py_any(py),
        GenericValue::ParameterExpression(exp) => exp.as_ref().clone().into_py_any(py),
        GenericValue::Circuit(py_object) => Ok(py_object.clone()),
        GenericValue::CircuitData(circuit_data) => {
            Ok(circuit_data.clone().into_py_quantum_circuit(py)?.unbind())
        }
        GenericValue::Modifier(py_object) => Ok(py_object.clone()),
        GenericValue::Range(py_range) => py_range.into_py_any(py),
        GenericValue::NumpyObject(py_object) => Ok(py_object.clone()),
        GenericValue::Tuple(values) => {
            let elements: Vec<Py<PyAny>> = values
                .iter()
                .map(py_convert_from_generic_value)
                .collect::<PyResult<_>>()?;
            PyTuple::new(py, &elements)?.into_py_any(py)
        }
        GenericValue::Register(reg_value) => match reg_value {
            ParamRegisterValue::Register(reg) => reg.clone().into_py_any(py),
            ParamRegisterValue::ShareableClbit(clbit) => clbit.clone().into_py_any(py),
        },
        GenericValue::BigInt(bigint) => bigint.clone().into_py_any(py),
        GenericValue::Duration(duration) => (*duration).into_py_any(py),
    })
}

// This functions packs an instruction parameter, which can be an arbitrary piece of data
// Not to be confused with Parameter, which is an atom of ParameterExpression
pub(crate) fn py_pack_param(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
    endian: Endian,
) -> PyResult<formats::GenericDataPack> {
    let value = py_convert_to_generic_value(py_object)?;
    let (type_key, data) = match endian {
        Endian::Big => serialize_generic_value(&value, qpy_data)?,
        Endian::Little => serialize_generic_value(&value.as_le(), qpy_data)?,
    };
    Ok(formats::GenericDataPack { type_key, data })
}

pub(crate) fn py_get_instruction_annotations(
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> PyResult<Option<formats::InstructionsAnnotationPack>> {
    Python::attach(|py| {
        if let OperationRef::Instruction(inst) = instruction.op.view() {
            let op = inst.instruction.bind(py);
            if op.is_instance(imports::CONTROL_FLOW_BOX_OP.get_bound(py))? {
                let annotations_iter = PyIterator::from_object(&op.getattr("annotations")?)?;
                let annotations: Vec<formats::InstructionAnnotationPack> = annotations_iter
                    .map(|annotation| {
                        let (namespace_index, payload) = qpy_data
                            .annotation_handler
                            .serialize(&annotation?.unbind())?;
                        Ok(formats::InstructionAnnotationPack {
                            namespace_index,
                            payload,
                        })
                    })
                    .collect::<PyResult<_>>()?;
                if !annotations.is_empty() {
                    return Ok(Some(formats::InstructionsAnnotationPack { annotations }));
                }
            }
        }
        Ok(None)
    })
}

pub(crate) fn py_pack_modifier(modifier: &Py<PyAny>) -> PyResult<formats::ModifierPack> {
    Python::attach(|py| {
        let modifier = modifier.bind(py);
        let module = py.import("qiskit.circuit.annotated_operation")?;
        if modifier.is_instance(&module.getattr("InverseModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: ModifierType::Inverse,
                num_ctrl_qubits: 0,
                ctrl_state: 0,
                power: 0.0,
            })
        } else if modifier.is_instance(&module.getattr("ControlModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: ModifierType::Control,
                num_ctrl_qubits: modifier.getattr("num_ctrl_qubits")?.extract::<u32>()?,
                ctrl_state: modifier.getattr("ctrl_state")?.extract::<u32>()?,
                power: 0.0,
            })
        } else if modifier.is_instance(&module.getattr("PowerModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: ModifierType::Power,
                num_ctrl_qubits: 0,
                ctrl_state: 0,
                power: modifier.getattr("power")?.extract::<f64>()?,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported modifier."))
        }
    })
}

pub(crate) fn py_unpack_modifier(packed_modifier: &formats::ModifierPack) -> PyResult<Py<PyAny>> {
    Python::attach(|py| match packed_modifier.modifier_type {
        ModifierType::Inverse => Ok(imports::INVERSE_MODIFIER.get_bound(py).call0()?.unbind()),
        ModifierType::Control => {
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
        ModifierType::Power => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "power"), packed_modifier.power)?;
            Ok(imports::POWER_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
    })
}
