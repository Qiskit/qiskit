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

// Methods for QPY serialization working directly with Python-based data
use hashbrown::HashMap;
use numpy::ToPyArray;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyTuple};

use qiskit_circuit::bit::{PyClassicalRegister, PyClbit, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use uuid::Uuid;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::formats;
use crate::params::{pack_param, pack_param_obj};
use crate::value::{
    circuit_instruction_types, dumps_py_value, get_circuit_type_key,
    pack_generic_data, serialize, DumpedPyValue, QPYWriteData,
};

use crate::circuit_writer::{
    pack_circuit, pack_custom_instructions, pack_instruction, pack_instructions, pack_standalone_vars
};

const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";

fn is_python_gate(py: Python, op: &PackedOperation, python_gate: &Bound<PyAny>) -> PyResult<bool> {
    match op.view() {
        OperationRef::Gate(pygate) => {
            if pygate.gate.bind(py).is_instance(python_gate)? {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

pub fn recognize_custom_operation(op: &PackedOperation, name: &String) -> PyResult<Option<String>> {
    Python::attach(|py| {
        let library = py.import("qiskit.circuit.library")?;
        let circuit_mod = py.import("qiskit.circuit")?;
        let controlflow = py.import("qiskit.circuit.controlflow")?;

        if (!library.hasattr(name)?
            && !circuit_mod.hasattr(name)?
            && !controlflow.hasattr(name)?
            && name != "Clifford")
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

pub fn get_python_gate_class<'a>(
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
    } else {
        Err(PyIOError::new_err(format!(
            "Gate class not found: {:?}",
            gate_class_name
        )))
    }
}

// serializes python metadata to JSON using a python JSON serializer
pub fn serialize_metadata(
    metadata_opt: &Option<Bound<PyAny>>,
    metadata_serializer: &Bound<PyAny>,
) -> PyResult<Bytes> {
    match metadata_opt {
        None => Ok(Bytes::new()),
        Some(metadata) => {
            let py = metadata.py();
            let json = py.import("json")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
            kwargs.set_item("cls", metadata_serializer)?;
            Ok(json
                .call_method("dumps", (metadata,), Some(&kwargs))?
                .extract::<String>()?
                .into())
        }
    }
}

// helper method to extract attribute from a py_object
pub fn getattr_or_none<'py>(
    py_object: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if py_object.hasattr(name)? {
        let attr = py_object.getattr(name)?;
        if attr.is_none() {
            Ok(None)
        } else {
            Ok(Some(attr))
        }
    } else {
        Ok(None)
    }
}

pub fn pack_py_registers(
    in_circ_regs: &Bound<PyAny>,
    bits: &Bound<PyList>,
) -> PyResult<Vec<formats::RegisterV4Pack>> {
    let py = in_circ_regs.py();
    let bitmap = PyDict::new(py);
    let out_circ_regs = PyList::new(py, Vec::<Py<PyAny>>::new())?;

    bits.iter()
        .enumerate()
        .try_for_each(|(index, bit)| -> PyResult<()> {
            bitmap.set_item(&bit, index)?;
            match getattr_or_none(&bit, "_register")? {
                None => Ok(()),
                Some(register) => {
                    if !(in_circ_regs.contains(&register).unwrap_or(false))
                        && !(out_circ_regs.contains(&register).unwrap_or(true))
                    {
                        out_circ_regs.append(register)?;
                    }
                    Ok(())
                }
            }
        })?;
    let mut result = Vec::new();
    in_circ_regs
        .cast::<PyList>()?
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(pack_py_register(&register, &bitmap, true)?);
            Ok(())
        })?;

    out_circ_regs
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(pack_py_register(&register, &bitmap, false)?);
            Ok(())
        })?;
    Ok(result)
}

fn pack_py_register(
    register: &Bound<PyAny>,
    bitmap: &Bound<PyDict>,
    is_in_circuit: bool,
) -> PyResult<formats::RegisterV4Pack> {
    let reg_name = register.getattr("name")?.extract::<String>()?;
    let reg_type = register
        .getattr("prefix")?
        .extract::<String>()?
        .into_bytes()[0];
    let mut standalone = true;
    let mut bit_indices: Vec<i64> = Vec::new();
    for (index, bit) in PyIterator::from_object(register)?.enumerate() {
        let bit_val = bit?;
        if !(register
            .rich_compare(bit_val.getattr("_register")?, pyo3::basic::CompareOp::Eq)?
            .is_truthy()?)
        {
            standalone = false;
        }
        match getattr_or_none(&bit_val, "_index")? {
            None => (),
            Some(value) => {
                if value.extract::<usize>()? != index {
                    standalone = false
                }
            }
        }

        if let Some(index) = bitmap.get_item(bit_val)? {
            bit_indices.push(index.extract::<i64>()?);
        } else {
            bit_indices.push(-1);
        }
    }
    let packed_reg = formats::RegisterV4Pack {
        register_type: reg_type,
        standalone: standalone as u8,
        in_circuit: is_in_circuit as u8,
        name: reg_name,
        bit_indices,
    };
    Ok(packed_reg)
}

fn pack_py_circuit_header(
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::CircuitHeaderV12Pack> {
    let py = circuit.py();
    let circuit_name = circuit.getattr(intern!(py, "name"))?.extract::<String>()?;
    let metadata = serialize_metadata(
        &Some(circuit.getattr(intern!(py, "metadata"))?),
        metadata_serializer,
    )?;
    let global_phase_data = DumpedPyValue::from(
        circuit.getattr(intern!(py, "global_phase"))?.unbind(),
        qpy_data,
    )?;
    let qregs = pack_py_registers(
        &circuit.getattr(intern!(py, "qregs"))?,
        circuit
            .getattr(intern!(py, "qubits"))?
            .cast::<PyList>()?,
    )?;
    let cregs = pack_py_registers(
        &circuit.getattr(intern!(py, "cregs"))?,
        circuit
            .getattr(intern!(py, "clbits"))?
            .cast::<PyList>()?,
    )?;
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: circuit
            .getattr(intern!(py, "num_qubits"))?
            .extract::<u32>()?,
        num_clbits: circuit
            .getattr(intern!(py, "num_clbits"))?
            .extract::<u32>()?,
        num_instructions: circuit
            .call_method0(intern!(py, "__len__"))?
            .extract::<u64>()?,
        num_vars: circuit
            .getattr(intern!(py, "num_identifiers"))?
            .extract::<u32>()?,
        circuit_name,
        global_phase_data,
        metadata,
        registers,
    };

    Ok(header)
}

fn pack_py_layout(circuit: &Bound<PyAny>) -> PyResult<formats::LayoutV2Pack> {
    if circuit.getattr(intern!(circuit.py(), "layout"))?.is_none() {
        Ok(formats::LayoutV2Pack {
            exists: 0,
            initial_layout_size: -1,
            input_mapping_size: -1,
            final_layout_size: -1,
            input_qubit_count: 0,
            extra_registers: Vec::new(),
            initial_layout_items: Vec::new(),
            input_mapping_items: Vec::new(),
            final_layout_items: Vec::new(),
        })
    } else {
        pack_py_custom_layout(circuit)
    }
}

fn pack_py_custom_layout(circuit: &Bound<PyAny>) -> PyResult<formats::LayoutV2Pack> {
    let layout = circuit.getattr("layout")?;
    let mut initial_layout_size = -1; // initial_size
    let py = circuit.py();
    let input_qubit_mapping = PyDict::new(py);
    let initial_layout_array = PyList::empty(py);
    let extra_registers = PyDict::new(py);

    let initial_layout = layout.getattr("initial_layout")?;
    let num_qubits: usize = circuit.getattr("num_qubits")?.extract()?;
    if !initial_layout.is_none() {
        initial_layout_size = initial_layout.call_method0("__len__")?.extract::<i32>()?;
        let layout_mapping = initial_layout.call_method0("get_physical_bits")?;
        for i in 0..num_qubits {
            let qubit = layout_mapping.get_item(i)?;
            input_qubit_mapping.set_item(&qubit, i)?;
            let register = qubit.getattr("_register")?;
            let index = qubit.getattr("_index")?;
            if !register.is_none() || !index.is_none() {
                if !circuit.getattr("qregs")?.contains(&register)? {
                    let extra_register_list = match extra_registers.get_item(&register)? {
                        Some(list) => list,
                        None => {
                            let new_list = PyList::empty(py);
                            extra_registers.set_item(&register, &new_list)?;
                            new_list.into_any()
                        }
                    };
                    extra_register_list.cast::<PyList>()?.append(qubit)?;
                }
                initial_layout_array.append((index, register))?;
            } else {
                initial_layout_array.append((py.None(), py.None()))?;
            }
        }
    }

    let mut input_mapping_size: i32 = -1; //input_qubit_size
    let mut input_qubit_mapping_array = PyList::new(py, Vec::<Bound<PyAny>>::new())?;
    let layout_input_qubit_mapping = layout.getattr("input_qubit_mapping")?;
    if !layout_input_qubit_mapping.is_none() {
        input_mapping_size = layout_input_qubit_mapping
            .call_method0("__len__")?
            .extract()?;
        input_qubit_mapping_array = PyList::new(
            py,
            std::iter::repeat(py.None())
                .take(input_mapping_size as usize)
                .collect::<Vec<_>>(),
        )?;
        let layout_mapping = initial_layout.call_method0("get_virtual_bits")?;
        for (qubit, index) in layout_input_qubit_mapping.cast::<PyDict>()? {
            let register = qubit.getattr("_register")?;
            if !register.is_none()
                && !qubit.getattr("_index")?.is_none()
                && !circuit.getattr("qregs")?.contains(&register)?
            {
                let extra_register_list = match extra_registers.get_item(&register)? {
                    Some(list) => list,
                    None => {
                        let new_list = PyList::empty(py);
                        extra_registers.set_item(&register, &new_list)?;
                        new_list.into_any()
                    }
                };
                extra_register_list.cast::<PyList>()?.append(&qubit)?;
            }
            input_qubit_mapping_array
                .set_item(index.extract()?, layout_mapping.get_item(&qubit)?)?;
        }
    }

    let mut final_layout_size: i32 = -1;
    let final_layout_array = PyList::empty(py);
    let final_layout = layout.getattr("final_layout")?;
    if !final_layout.is_none() {
        final_layout_size = final_layout.call_method0("__len__")?.extract()?;
        let final_layout_physical = final_layout.call_method0("get_physical_bits")?;
        for i in 0..num_qubits {
            let virtual_bit = final_layout_physical.cast::<PyDict>()?.get_item(i)?;
            final_layout_array.append(
                circuit
                    .call_method1("find_bit", (virtual_bit,))?
                    .getattr("index")?,
            )?;
        }
    }

    let input_qubit_count: i32 = if layout.getattr("_input_qubit_count")?.is_none() {
        -1
    } else {
        layout.getattr("_input_qubit_count")?.extract()?
    };

    let mut bits = Vec::new();
    for register_bit_list in extra_registers.values() {
        for x in register_bit_list.cast::<PyList>()? {
            bits.push(x);
        }
    }
    let extra_registers = pack_py_registers(&extra_registers.keys(), &PyList::new(py, bits)?)?;
    let mut initial_layout_items = Vec::with_capacity(initial_layout_size.max(0) as usize);
    for item in initial_layout_array {
        let tuple = item.cast::<PyTuple>()?;
        let index = tuple.get_item(0)?;
        let register = tuple.get_item(1)?;
        let reg_name_bytes = if !register.is_none() {
            Some(register.getattr("name")?.extract::<String>()?)
        } else {
            None
        };
        let index_value = if index.is_none() {
            -1
        } else {
            index.extract::<i32>()?
        };
        let (register_name, register_name_length) = reg_name_bytes
            .as_ref()
            .map(|name| (name.clone(), name.len() as i32))
            .unwrap_or((String::new(), -1));
        initial_layout_items.push(formats::InitialLayoutItemV2Pack {
            index_value,
            register_name_length,
            register_name,
        });
    }

    let mut input_mapping_items = Vec::with_capacity(input_mapping_size.max(0) as usize);
    for i in &input_qubit_mapping_array {
        input_mapping_items.push(i.extract::<u32>()?);
        // buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    let mut final_layout_items = Vec::with_capacity(final_layout_size.max(0) as usize);
    for i in &final_layout_array {
        final_layout_items.push(i.extract::<u32>()?);
        // buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    Ok(formats::LayoutV2Pack {
        exists: true as u8,
        initial_layout_size,
        input_mapping_size,
        final_layout_size,
        input_qubit_count,
        extra_registers,
        initial_layout_items,
        input_mapping_items,
        final_layout_items,
    })
}

pub fn pack_py_circuit(
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<formats::QPYFormatV15> {
    let py = circuit.py();
    circuit.getattr("data")?; // in case _data is lazily generated in python
    let mut circuit_data = circuit.getattr("_data")?.extract::<CircuitData>()?;
    let mut standalone_var_indices: HashMap<u128, u16> = HashMap::new();
    let standalone_vars = pack_standalone_vars(&circuit_data, version, &mut standalone_var_indices)?;
    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYWriteData {
        version,
        _use_symengine: use_symengine,
        clbits: &circuit_data.clbits().clone(), // we need to clone since circuit_data might change when serializing custom instructions, explicitly creating the inner instructions
        standalone_var_indices: HashMap::new(),
        annotation_handler,
    };
    let header = pack_py_circuit_header(circuit, metadata_serializer, &qpy_data)?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack { num_cals: 0 };
    let (instructions, mut custom_instructions_hash) =
        pack_instructions(&circuit_data, &mut qpy_data)?;
    let custom_instructions = pack_custom_instructions(
        &mut custom_instructions_hash,
        &mut circuit_data,
        &mut qpy_data,
    )?;
    let layout = pack_py_layout(circuit)?;
    let state_headers: Vec<formats::AnnotationStateHeaderPack> = qpy_data
        .annotation_handler
        .dump_serializers()?
        .into_iter()
        .map(|(namespace, state)| formats::AnnotationStateHeaderPack { namespace, state })
        .collect();

    let annotation_headers = formats::AnnotationHeaderStaticPack { state_headers };
    Ok(formats::QPYFormatV15 {
        header,
        standalone_vars,
        annotation_headers,
        custom_instructions,
        instructions,
        calibrations,
        layout,
    })
}

fn pack_sparse_pauli_op(
    operator: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::SparsePauliOpListElemPack> {
    let op_as_np_list = operator.call_method1("to_list", (true,))?;
    let (_, data) = dumps_py_value(op_as_np_list.unbind(), qpy_data)?;
    Ok(formats::SparsePauliOpListElemPack { data })
}

fn pack_pauli_evolution_gate(
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

    let (time_type, time_data) = dumps_py_value(evolution_gate.getattr("time")?.unbind(), qpy_data)?;
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

pub fn gate_class_name(op: &PackedOperation) -> PyResult<String> {
    Python::attach(|py| {
        let name = match op.view() {
            // getting __name__ for standard gates and instructions should
            // eventually be replaced with a Rust-side mapping
            OperationRef::StandardGate(gate) => gate
                .get_gate_class(py)?
                .bind(py)
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
            OperationRef::StandardInstruction(inst) => match inst {
                StandardInstruction::Measure => imports::MEASURE
                    .get_bound(py)
                    .getattr(intern!(py, "__name__"))?,
                StandardInstruction::Delay(_) => imports::DELAY
                    .get_bound(py)
                    .getattr(intern!(py, "__name__"))?,
                StandardInstruction::Barrier(_) => imports::BARRIER
                    .get_bound(py)
                    .getattr(intern!(py, "__name__"))?,
                StandardInstruction::Reset => imports::RESET
                    .get_bound(py)
                    .getattr(intern!(py, "__name__"))?,
            }
            .extract::<String>(),
            OperationRef::Gate(pygate) => pygate
                .gate
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
                .operation
                .bind(py)
                .getattr(intern!(py, "__class__"))?
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
        }?;
        Ok(name)
    })
}

pub fn get_instruction_params(
    instruction: &PackedInstruction,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::PackedParam>> {
    // The instruction params we store are about being able to reconstruct the objects; they don't
    // necessarily need to match one-to-one to the `params` field.
    Python::attach(|py| {
        if let OperationRef::Instruction(inst) = instruction.op.view() {
            if inst
                .instruction
                .bind(py)
                .is_instance(imports::CONTROL_FLOW_SWITCH_CASE_OP.get_bound(py))?
            {
                let op = inst.instruction.bind(py);
                let target = op.getattr("target")?;
                let cases = op.call_method0("cases_specifier")?;
                let cases_tuple = imports::BUILTIN_TUPLE.get_bound(py).call1((cases,))?;
                return Ok(vec![
                    pack_param(&target, qpy_data)?,
                    pack_param(&cases_tuple, qpy_data)?,
                ]);
            }
            if inst
                .instruction
                .bind(py)
                .is_instance(imports::CONTROL_FLOW_BOX_OP.get_bound(py))?
            {
                let op = inst.instruction.bind(py);
                let first_block = op
                    .getattr("blocks")?
                    .try_iter()?
                    .next()
                    .transpose()?
                    .ok_or_else(|| PyValueError::new_err("No blocks in box control flow op"))?;
                let duration = op.getattr("duration")?;
                let unit = op.getattr("unit")?;
                return Ok(vec![
                    pack_param(&first_block, qpy_data)?,
                    pack_param(&duration, qpy_data)?,
                    pack_param(&unit, qpy_data)?,
                ]);
            }
        }

        if let OperationRef::Operation(op) = instruction.op.view() {
            if op
                .operation
                .bind(py)
                .is_instance(imports::CLIFFORD.get_bound(py))?
            {
                let op = op.operation.bind(py);
                let tableau = op.getattr("tableau")?;
                return Ok(vec![pack_param(&tableau, qpy_data)?]);
            }
            if op
                .operation
                .bind(py)
                .is_instance(imports::ANNOTATED_OPERATION.get_bound(py))?
            {
                let op = op.operation.bind(py);
                let modifiers = op.getattr("modifiers")?;
                return modifiers
                    .try_iter()?
                    .map(|modifier| pack_param(&modifier?, qpy_data))
                    .collect::<PyResult<_>>();
            }
        }

        // elif isinstance(instruction.operation, AnnotatedOperation):
        //instruction_params = instruction.operation.modifiers
        if let OperationRef::Unitary(unitary) = instruction.op.view() {
            // unitary gates are special since they are uniquely determined by a matrix, which is not
            // a "parameter", strictly speaking, but is treated as such when serializing

            // until we change the QPY version or verify we get the exact same result,
            // we translate the matrix to numpy and then serialize it like python does
            let out_array = match &unitary.array {
                ArrayType::NDArray(arr) => arr.to_pyarray(py),
                ArrayType::OneQ(arr) => arr.to_pyarray(py),
                ArrayType::TwoQ(arr) => arr.to_pyarray(py),
            };
            return Ok(vec![pack_param(&out_array, qpy_data)?]);
        }
        instruction
            .params_view()
            .iter()
            .map(|x| pack_param_obj(py, x, qpy_data))
            .collect::<PyResult<_>>()
    })
}

pub fn get_instruction_annotations(
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
                        let (namespace_index, payload) =
                            qpy_data.annotation_handler.serialize(&annotation?)?;
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

pub fn pack_custom_instruction(
    name: &String,
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    new_instructions_list: &mut Vec<String>,
    circuit_data: &mut CircuitData,
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CustomCircuitInstructionDefPack> {
    Python::attach(|py| {
        let operation = custom_instructions_hash.get(name).ok_or_else(|| {
            PyValueError::new_err(format!("Could not find operation data for {}", name))
        })?;
        let gate_type = get_circuit_type_key(operation)?;
        let mut has_definition = false;
        let mut data: Bytes = Bytes::new();
        let mut num_ctrl_qubits = 0;
        let mut ctrl_state = 0;
        let mut base_gate: Bound<PyAny> = py.None().bind(py).clone();
        let mut base_gate_raw: Bytes = Bytes::new();

        if gate_type == circuit_instruction_types::PAULI_EVOL_GATE {
            if let OperationRef::Gate(gate) = operation.view() {
                has_definition = true;
                data = serialize(&pack_pauli_evolution_gate(gate.gate.bind(py), qpy_data)?);
            }
        } else if gate_type == circuit_instruction_types::CONTROLLED_GATE {
            // For ControlledGate, we have to access and store the private `_definition` rather than the
            // public one, because the public one is mutated to include additional logic if the control
            // state is open, and the definition setter (during a subsequent read) uses the "fully
            // excited" control definition only.
            if let OperationRef::Gate(pygate) = operation.view() {
                has_definition = true;
                // Build internal definition to support overloaded subclasses by
                // calling definition getter on object
                let gate = pygate.gate.bind(py);
                gate.getattr("definition")?; // this creates the _definition field
                data = serialize(&pack_circuit(
                    gate.getattr("_definition")?.extract()?,
                    py.None().bind(py),
                    false,
                    qpy_data.version,
                    qpy_data.annotation_handler.annotation_factories,
                )?);
                num_ctrl_qubits = gate.getattr("num_ctrl_qubits")?.extract::<u32>()?;
                ctrl_state = gate.getattr("ctrl_state")?.extract::<u32>()?;
                base_gate = gate.getattr("base_gate")?.clone();
            }
        } else if gate_type == circuit_instruction_types::ANNOTATED_OPERATION {
            if let OperationRef::Operation(operation) = operation.view() {
                has_definition = false; // just making sure
                base_gate = operation.operation.bind(py).getattr("base_op")?.clone();
            }
        } else {
            match operation.view() {
                // all-around catch for "operation" field; should be easier once we switch from python to rust
                OperationRef::Gate(pygate) => {
                    let gate = pygate.gate.bind(py);
                    match getattr_or_none(gate, "definition")? {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_py_circuit(
                                &definition,
                                py.None().bind(py),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                OperationRef::Instruction(pyinst) => {
                    let inst = pyinst.instruction.bind(py);
                    match getattr_or_none(inst, "definition")? {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_py_circuit(
                                &definition,
                                py.None().bind(py),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                OperationRef::Operation(pyoperation) => {
                    let operation = pyoperation.operation.bind(py);
                    match getattr_or_none(operation, "definition")? {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_py_circuit(
                                &definition,
                                py.None().bind(py),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                _ => (),
            }
        }
        let num_qubits = operation.num_qubits();
        let num_clbits = operation.num_clbits();
        if !base_gate.is_none() {
            let instruction =
                circuit_data.pack(py, &CircuitInstruction::py_new(&base_gate, None, None)?)?;
            base_gate_raw = serialize(&pack_instruction(
                &instruction,
                circuit_data,
                custom_instructions_hash,
                new_instructions_list,
                qpy_data,
            )?);
        }
        Ok(formats::CustomCircuitInstructionDefPack {
            gate_type,
            num_qubits,
            num_clbits,
            custom_definition: has_definition as u8,
            num_ctrl_qubits,
            ctrl_state,
            name: name.to_string(),
            data,
            base_gate_raw,
        })
    })
}

fn dumps_register(register: Bound<PyAny>, circuit_data: &CircuitData) -> PyResult<Bytes> {
    let py = register.py();
    if register.is_instance_of::<PyClassicalRegister>() {
        Ok(register.getattr("name")?.extract::<String>()?.into())
    } else if register.is_instance_of::<PyClbit>() {
        let key = &register.extract::<ShareableClbit>()?;
        let name = circuit_data
            .get_clbit_indices(py)
            .bind(py)
            .get_item(key)?
            .ok_or(PyErr::new::<PyValueError, _>("Clbit not found"))?
            .getattr("index")?
            .str()?;
        let mut bytes: Bytes = Bytes(Vec::with_capacity(name.len()? + 1));
        bytes.push(0u8);
        bytes.extend_from_slice(name.extract::<String>()?.as_bytes());
        Ok(bytes)
    } else {
        Ok(Bytes::new())
    }
}

pub fn get_condition_data_from_inst(
    inst: &Py<PyAny>,
    circuit_data: &CircuitData,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ConditionPack> {
    Python::attach(|py| match getattr_or_none(inst.bind(py), "_condition")? {
        None => Ok(formats::ConditionPack {
            key: formats::condition_types::NONE,
            register_size: 0u16,
            value: 0i64,
            data: formats::ConditionData::None,
        }),
        Some(condition) => {
            if condition.extract::<classical::expr::Expr>().is_ok() {
                let expression = pack_generic_data(&condition, qpy_data)?;
                Ok(formats::ConditionPack {
                    key: formats::condition_types::EXPRESSION,
                    register_size: 0u16,
                    value: 0i64,
                    data: formats::ConditionData::Expression(expression),
                })
            } else if condition.is_instance_of::<PyTuple>() {
                let key = formats::condition_types::TWO_TUPLE;
                let value = condition
                    .cast::<PyTuple>()?
                    .get_item(1)?
                    .extract::<i64>()?;
                let register =
                    dumps_register(condition.cast::<PyTuple>()?.get_item(0)?, circuit_data)?;
                Ok(formats::ConditionPack {
                    key,
                    register_size: register.len() as u16,
                    value,
                    data: formats::ConditionData::Register(register),
                })
            } else {
                Err(PyValueError::new_err(
                    "Expression handling not implemented for get_condition_data_from_inst",
                ))
            }
        }
    })
}

// pub fn unpack_py_parameter_expression(
//     py: Python,
//     parameter_expression: formats::ParameterExpressionPack,
//     qpy_data: &mut QPYReadData,
// ) -> PyResult<Py<PyAny>> {
//     let mut param_uuid_map: HashMap<[u8; 16], Py<PyAny>> = HashMap::new();
//     let mut name_map: HashMap<String, Py<PyAny>> = HashMap::new();

//     let mut stack: Vec<Py<PyAny>> = Vec::new();
//     for item in &parameter_expression.symbol_table_data {
//         let (symbol_uuid, symbol, value) = match item {
//             formats::ParameterExpressionSymbolPack::ParameterExpression(_) => {
//                 continue;
//             }
//             formats::ParameterExpressionSymbolPack::Parameter(symbol_pack) => {
//                 let symbol = unpack_parameter(py, &symbol_pack.symbol_data)?;
//                 let value = if symbol_pack.value_key != parameter_tags::PARAMETER {
//                     let dumped_value = DumpedPyValue {
//                         data_type: symbol_pack.value_key,
//                         data: symbol_pack.value_data.clone(),
//                     };
//                     dumped_value.to_python(py, qpy_data)?
//                 } else {
//                     symbol.clone()
//                 };
//                 (symbol_pack.symbol_data.uuid, symbol, value)
//             }
//             formats::ParameterExpressionSymbolPack::ParameterVector(symbol_pack) => {
//                 let symbol = unpack_parameter_vector(py, &symbol_pack.symbol_data, qpy_data)?;
//                 let value = if symbol_pack.value_key != parameter_tags::PARAMETER_VECTOR {
//                     let dumped_value = DumpedPyValue {
//                         data_type: symbol_pack.value_key,
//                         data: symbol_pack.value_data.clone(),
//                     };
//                     dumped_value.to_python(py, qpy_data)?
//                 } else {
//                     symbol.clone()
//                 };
//                 (symbol_pack.symbol_data.uuid, symbol, value)
//             }
//         };
//         param_uuid_map.insert(symbol_uuid, value.clone());
//         // name_map should only be used for version < 15
//         name_map.insert(
//             value
//                 .bind(py)
//                 .call_method0("__str__")?
//                 .extract::<String>()?,
//             symbol,
//         );
//     }
//     let parameter_expression_data = deserialize_vec::<formats::ParameterExpressionElementPack>(
//         &parameter_expression.expression_data,
//     )?;
//     for element in parameter_expression_data {
//         let opcode = if let formats::ParameterExpressionElementPack::Substitute(subs) = element {
//             // we construct a pydictionary describing the substitution and letting the python Parameter class handle it
//             let subs_mapping = PyDict::new(py);
//             let mapping_pack = deserialize::<formats::MappingPack>(&subs.mapping_data)?.0;
//             for item in mapping_pack.items {
//                 let key_uuid: [u8; 16] = (&item.key_bytes).try_into()?;
//                 let value = DumpedPyValue {
//                     data_type: item.item_type,
//                     data: item.item_bytes.clone(),
//                 };
//                 let key = param_uuid_map.get(&key_uuid).ok_or_else(|| {
//                     PyValueError::new_err(format!("Parameter UUID not found: {:?}", &key_uuid))
//                 })?;
//                 subs_mapping.set_item(key, value.to_python(py, qpy_data)?)?;
//             }
//             stack.push(subs_mapping.unbind().as_any().clone());
//             15 // return substitution opcode
//         } else {
//             let (opcode, op) = unpack_parameter_expression_standard_op(element)?;
//             // LHS
//             match op.lhs_type {
//                 parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
//                     if let Some(value) = param_uuid_map.get(&op.lhs) {
//                         stack.push(value.clone());
//                     } else {
//                         return Err(PyValueError::new_err(format!(
//                             "Parameter UUID not found: {:?}",
//                             op.lhs
//                         )));
//                     }
//                 }
//                 parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
//                     if let Some(value) = unpack_parameter_replay_entry(py, op.lhs_type, op.lhs)? {
//                         stack.push(value);
//                     }
//                 }
//                 parameter_tags::NULL => (), // pass
//                 parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
//                 _ => {
//                     return Err(PyValueError::new_err(format!(
//                         "Unknown ParameterExpression operation type: {}",
//                         op.lhs_type
//                     )))
//                 }
//             }
//             // RHS
//             match op.rhs_type {
//                 parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
//                     if let Some(value) = param_uuid_map.get(&op.rhs) {
//                         stack.push(value.clone());
//                     } else {
//                         return Err(PyValueError::new_err(format!(
//                             "Parameter UUID not found: {:?}",
//                             op.rhs
//                         )));
//                     }
//                 }
//                 parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
//                     if let Some(value) = unpack_parameter_replay_entry(py, op.rhs_type, op.rhs)? {
//                         stack.push(value);
//                     }
//                 }
//                 parameter_tags::NULL => (), // pass
//                 parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
//                 _ => {
//                     return Err(PyTypeError::new_err(format!(
//                         "Unknown ParameterExpression operation type: {}",
//                         op.rhs_type
//                     )))
//                 }
//             }
//             if opcode == 255 {
//                 continue;
//             }
//             opcode
//         };
//         let method_str = op_code_to_method(opcode)?;

//         if [0, 1, 2, 3, 4, 13, 15, 18, 19, 20].contains(&opcode) {
//             let rhs = stack.pop().ok_or(PyTypeError::new_err(
//                 "Stack underflow while parsing parameter expression",
//             ))?;
//             let lhs = stack.pop().ok_or(PyTypeError::new_err(
//                 "Stack underflow while parsing parameter expression",
//             ))?;
//             // Reverse ops for commutative ops, which are add, mul (0 and 2 respectively)
//             // op codes 13 and 15 can never be reversed and 18, 19, 20
//             // are the reversed versions of non-commuative operations
//             // so 1, 3, 4 and 18, 19, 20 handle this explicitly.
//             if [0, 2].contains(&opcode)
//                 && !lhs
//                     .bind(py)
//                     .is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))?
//                 && rhs
//                     .bind(py)
//                     .is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))?
//             {
//                 let method_str = match &opcode {
//                     0 => "__radd__",
//                     2 => "__rmul__",
//                     _ => method_str,
//                 };
//                 stack.push(rhs.getattr(py, method_str)?.call1(py, (lhs,))?);
//             } else {
//                 stack.push(lhs.getattr(py, method_str)?.call1(py, (rhs,))?);
//             }
//         } else {
//             // unary op
//             let lhs = stack.pop().ok_or(PyValueError::new_err(
//                 "Stack underflow while parsing parameter expression",
//             ))?;
//             stack.push(lhs.getattr(py, method_str)?.call0(py)?);
//         }
//     }

//     let result = stack.pop().ok_or(PyValueError::new_err(
//         "Stack underflow while parsing parameter expression",
//     ))?;
//     Ok(result)
// }