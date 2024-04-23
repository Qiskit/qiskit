// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(clippy::too_many_arguments)]

use std::hash::{Hash, Hasher};

use hashbrown::HashSet;
use indexmap::IndexMap;
use pyo3::{
    exceptions::{PyAttributeError, PyIndexError, PyKeyError},
    prelude::*,
    pyclass,
    types::{IntoPyDict, PyDict, PyList, PyTuple, PyType},
};

use self::exceptions::{QiskitError, TranspilerError};

// This struct allows qargs and any vec to become hashable
#[derive(Eq, PartialEq, Clone, Debug)]
struct HashableVec<T> {
    pub vec: Vec<T>,
}

impl<T: Hash> Hash for HashableVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for qarg in self.vec.iter() {
            qarg.hash(state);
        }
    }
}

impl<T: ToPyObject> IntoPy<PyObject> for HashableVec<T> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyTuple::new_bound(py, self.vec).to_object(py)
    }
}

impl<'a, 'b: 'a, T> FromPyObject<'b> for HashableVec<T>
where
    Vec<T>: FromPyObject<'a>,
{
    fn extract(ob: &'b PyAny) -> PyResult<Self> {
        Ok(Self {
            vec: ob.extract::<Vec<T>>()?,
        })
    }

    fn extract_bound(ob: &Bound<'b, PyAny>) -> PyResult<Self> {
        Self::extract(ob.clone().into_gil_ref())
    }
}

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.exceptions, QiskitError}
    import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
    import_exception_bound! {qiskit.providers.exceptions, BackendPropertyError}
}

// Helper function to import inspect.isclass from python.
fn isclass(py: Python<'_>, object: &Bound<PyAny>) -> PyResult<bool> {
    let inspect_module: Bound<PyModule> = py.import_bound("inspect")?;
    let is_class_method: Bound<PyAny> = inspect_module.getattr("isclass")?;
    is_class_method.call1((object,))?.extract::<bool>()
}

// Helper function to import standard gate name mapping from python.
fn get_standard_gate_name_mapping(py: Python<'_>) -> PyResult<IndexMap<String, Bound<PyAny>>> {
    let inspect_module: Bound<PyModule> =
        py.import_bound("qiskit.circuit.library.standard_gates")?;
    let is_class_method: Bound<PyAny> = inspect_module.getattr("get_standard_gate_name_mapping")?;
    is_class_method
        .call0()?
        .extract::<IndexMap<String, Bound<PyAny>>>()
}

// Subclassable or Python Wrapping.
#[pyclass(subclass, module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
    #[pyo3(get)]
    _calibration: Option<PyObject>,
}

#[pymethods]
impl InstructionProperties {
    /**
     A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
     */
    #[new]
    #[pyo3(text_signature = "(/, duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,)")]
    pub fn new(
        py: Python<'_>,
        duration: Option<f64>,
        error: Option<f64>,
        calibration: Option<Bound<PyAny>>,
    ) -> Self {
        let mut instruction_prop = InstructionProperties {
            error,
            duration,
            _calibration: None,
        };
        if let Some(calibration) = calibration {
            let _ = instruction_prop.set_calibration(py, calibration);
        }
        instruction_prop
    }

    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> Option<PyObject> {
        /*
         The pulse representation of the instruction.

        .. note::

            This attribute always returns a Qiskit pulse program, but it is internally
            wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
            and to uniformly handle different data representation,
            for example, un-parsed Pulse Qobj JSON that a backend provider may provide.

            This value can be overridden through the property setter in following manner.
            When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
            always treated as a user-defined (custom) calibration and
            the transpiler may automatically attach the calibration data to the output circuit.
            This calibration data may appear in the wire format as an inline calibration,
            which may further update the backend standard instruction set architecture.

            If you are a backend provider who provides a default calibration data
            that is not needed to be attached to the transpiled quantum circuit,
            you can directly set :class:`.CalibrationEntry` instance to this attribute,
            in which you should set :code:`user_provided=False` when you define
            calibration data for the entry. End users can still intentionally utilize
            the calibration data, for example, to run pulse-level simulation of the circuit.
            However, such entry doesn't appear in the wire format, and backend must
            use own definition to compile the circuit down to the execution format.
         */
        match &self._calibration {
            Some(calibration) => calibration.call_method0(py, "get_schedule").ok(),
            None => None,
        }
    }

    #[setter]
    pub fn set_calibration(&mut self, py: Python<'_>, calibration: Bound<PyAny>) -> PyResult<()> {
        let module = py.import_bound("qiskit.pulse.schedule")?;
        // Import Schedule and ScheduleBlock types.
        let schedule_type = module.getattr("Schedule")?;
        let schedule_type = schedule_type.downcast::<PyType>()?;
        let schedule_block_type = module.getattr("ScheduleBlock")?;
        let schedule_block_type = schedule_block_type.downcast::<PyType>()?;
        if calibration.is_instance(schedule_block_type)?
            || calibration.is_instance(schedule_type)?
        {
            // Import the calibration_entries module
            let calibration_entries = py.import_bound("qiskit.pulse.calibration_entries")?;
            // Import the schedule def class.
            let schedule_def = calibration_entries.getattr("ScheduleDef")?;
            // Create a ScheduleDef instance.
            let new_entry: Bound<PyAny> = schedule_def.call0()?;
            // Definethe schedule, make sure it is user provided.
            let args = (calibration,);
            let kwargs = [("user_provided", true)].into_py_dict_bound(py);
            new_entry.call_method("define", args, Some(&kwargs))?;
            self._calibration = Some(new_entry.unbind());
        } else {
            self._calibration = Some(calibration.unbind());
        }
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut output = "InstructionProperties(".to_owned();
        if let Some(duration) = self.duration {
            output.push_str("duration=");
            output.push_str(duration.to_string().as_str());
            output.push_str(", ");
        } else {
            output.push_str("duration=None, ");
        }

        if let Some(error) = self.error {
            output.push_str("error=");
            output.push_str(error.to_string().as_str());
            output.push_str(", ");
        } else {
            output.push_str("error=None, ");
        }

        if let Some(calibration) = self.get_calibration(py) {
            output.push_str(
                format!(
                    "calibration={:?})",
                    calibration
                        .call_method0(py, "__str__")?
                        .extract::<String>(py)?
                )
                .as_str(),
            );
        } else {
            output.push_str("calibration=None)");
        }
        Ok(output)
    }
}

type GateMapType =
    IndexMap<String, Option<IndexMap<Option<HashableVec<u32>>, Option<InstructionProperties>>>>;
type TargetValue = Option<IndexMap<Option<HashableVec<u32>>, Option<InstructionProperties>>>;
type ErrorDictType<'a> = IndexMap<String, IndexMap<HashableVec<u32>, Bound<'a, PyAny>>>;
#[pyclass(mapping, subclass, module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
pub struct Target {
    #[pyo3(get, set)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub num_qubits: Option<usize>,
    #[pyo3(get, set)]
    pub dt: Option<f64>,
    #[pyo3(get, set)]
    pub granularity: i32,
    #[pyo3(get, set)]
    pub min_length: usize,
    #[pyo3(get, set)]
    pub pulse_alignment: i32,
    #[pyo3(get, set)]
    pub acquire_alignment: i32,
    #[pyo3(get, set)]
    pub qubit_properties: Vec<PyObject>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Vec<Vec<usize>>,
    // Maybe convert PyObjects into rust representations of Instruction and Data
    #[pyo3(get)]
    gate_map: GateMapType,
    #[pyo3(get)]
    gate_name_map: IndexMap<String, PyObject>,
    global_operations: IndexMap<usize, HashSet<String>>,
    #[pyo3(get)]
    qarg_gate_map: IndexMap<Option<HashableVec<u32>>, Option<HashSet<String>>>,
    #[pyo3(get, set)]
    instruction_durations: Option<PyObject>,
    instruction_schedule_map: Option<PyObject>,
    #[pyo3(get, set)]
    coupling_graph: Option<PyObject>,
    non_global_strict_basis: Option<Vec<String>>,
    non_global_basis: Option<Vec<String>>,
}

#[pymethods]
impl Target {
    #[new]
    #[pyo3(text_signature = "(/, description=None,
        num_qubits=0,
        dt=None,
        granularity=1,
        min_length=1,
        pulse_alignment=1,
        acquire_alignment=1,
        qubit_properties=None,
        concurrent_measurements=None,)")]
    fn new(
        description: Option<String>,
        num_qubits: Option<usize>,
        dt: Option<f64>,
        granularity: Option<i32>,
        min_length: Option<usize>,
        pulse_alignment: Option<i32>,
        acquire_alignment: Option<i32>,
        qubit_properties: Option<Vec<PyObject>>,
        concurrent_measurements: Option<Vec<Vec<usize>>>,
    ) -> Self {
        Target {
            description,
            num_qubits,
            dt,
            granularity: granularity.unwrap_or(1),
            min_length: min_length.unwrap_or(1),
            pulse_alignment: pulse_alignment.unwrap_or(1),
            acquire_alignment: acquire_alignment.unwrap_or(0),
            qubit_properties: qubit_properties.unwrap_or(Vec::new()),
            concurrent_measurements: concurrent_measurements.unwrap_or(Vec::new()),
            gate_map: IndexMap::new(),
            gate_name_map: IndexMap::new(),
            global_operations: IndexMap::new(),
            qarg_gate_map: IndexMap::new(),
            coupling_graph: None,
            instruction_durations: None,
            instruction_schedule_map: None,
            non_global_basis: None,
            non_global_strict_basis: None,
        }
    }

    #[pyo3(text_signature = "(/, instruction, properties=None, name=None")]
    fn add_instruction(
        &mut self,
        py: Python<'_>,
        instruction: &Bound<PyAny>,
        properties: Option<IndexMap<Option<HashableVec<u32>>, Option<InstructionProperties>>>,
        name: Option<String>,
    ) -> PyResult<()> {
        // Unwrap instruction name
        let instruction_name: String;
        let mut properties = properties;
        if !isclass(py, instruction)? {
            if let Some(name) = name {
                instruction_name = name;
            } else {
                instruction_name = instruction.getattr("name")?.extract::<String>()?;
            }
        } else {
            if let Some(name) = name {
                instruction_name = name;
            } else {
                return Err(TranspilerError::new_err(
                    "A name must be specified when defining a supported global operation by class",
                ));
            }
            if properties.is_some() {
                return Err(TranspilerError::new_err(
                    "An instruction added globally by class can't have properties set.",
                ));
            }
        }
        if properties.is_none() {
            properties = Some(IndexMap::from_iter([(None, None)].into_iter()));
        }
        if self.gate_map.contains_key(&instruction_name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {:?} is already in the target",
                instruction_name
            )));
        }
        self.gate_name_map
            .insert(instruction_name.clone(), instruction.clone().unbind());
        let mut qargs_val: IndexMap<Option<HashableVec<u32>>, Option<InstructionProperties>> =
            IndexMap::new();
        if isclass(py, instruction)? {
            qargs_val = IndexMap::from_iter([(None, None)].into_iter());
        } else if let Some(properties) = properties {
            let inst_num_qubits = instruction.getattr("num_qubits")?.extract::<usize>()?;
            if properties.contains_key(&None) {
                self.global_operations
                    .entry(inst_num_qubits)
                    .and_modify(|e| {
                        e.insert(instruction_name.clone());
                    })
                    .or_insert(HashSet::from_iter([instruction_name.clone()].into_iter()));
            }
            for qarg in properties.keys().cloned() {
                if let Some(qarg) = qarg.clone() {
                    if qarg.vec.len() != inst_num_qubits {
                        return Err(TranspilerError::new_err(
                            format!("The number of qubits for {instruction} does not match the number of qubits in the properties dictionary: {:?}", qarg.vec)
                        ));
                    }
                    self.num_qubits = Some(
                        self.num_qubits
                            .unwrap_or_default()
                            .max(qarg.vec.iter().cloned().fold(0, u32::max) as usize + 1),
                    );
                }
                qargs_val.insert(qarg.clone(), properties[&qarg].clone());
                self.qarg_gate_map
                    .entry(qarg)
                    .and_modify(|e| {
                        if let Some(e) = e {
                            e.insert(instruction_name.clone());
                        }
                    })
                    .or_insert(Some(HashSet::from([instruction_name.clone()])));
            }
        }
        self.gate_map.insert(instruction_name, Some(qargs_val));
        self.coupling_graph = None;
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        self.non_global_basis = None;
        self.non_global_strict_basis = None;
        Ok(())
    }

    #[pyo3(text_signature = "(/, instruction, qargs, properties)")]
    fn update_instruction_properties(
        &mut self,
        _py: Python<'_>,
        instruction: String,
        qargs: Option<HashableVec<u32>>,
        properties: Option<InstructionProperties>,
    ) -> PyResult<()> {
        /* Update the property object for an instruction qarg pair already in the Target

        Args:
            instruction (str): The instruction name to update
            qargs (tuple): The qargs to update the properties of
            properties (InstructionProperties): The properties to set for this instruction
        Raises:
            KeyError: If ``instruction`` or ``qarg`` are not in the target */

        // For debugging
        if !self.gate_map.contains_key(&instruction) {
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        if let Some(gate_map_instruction) = self.gate_map[&instruction].as_ref() {
            if !gate_map_instruction.contains_key(&qargs) {
                return Err(PyKeyError::new_err(format!(
                    "Provided qarg {:?} not in this Target for {:?}.",
                    &qargs.unwrap_or(HashableVec { vec: vec![] }).vec,
                    &instruction
                )));
            }
        }
        if let Some(Some(q_vals)) = self.gate_map.get_mut(&instruction) {
            if let Some(q_vals) = q_vals.get_mut(&qargs) {
                *q_vals = properties;
            }
        }
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
    }

    #[pyo3(text_signature = "(/, inst_map, inst_name_map=None, error_dict=None")]
    fn update_from_instruction_schedule_map(
        &mut self,
        py: Python<'_>,
        inst_map: &Bound<PyAny>,
        inst_name_map: Option<IndexMap<String, Bound<PyAny>>>,
        error_dict: Option<ErrorDictType>,
    ) -> PyResult<()> {
        let get_calibration = inst_map.getattr("_get_calibration_entry")?;
        // Expand name mapping with custom gate name provided by user.
        let mut qiskit_inst_name_map = get_standard_gate_name_mapping(py)?;

        if let Some(inst_name_map) = inst_name_map.as_ref() {
            for (key, value) in inst_name_map.iter() {
                qiskit_inst_name_map.insert(key.to_owned(), value.to_owned());
            }
        }

        let inst_map_instructions = inst_map.getattr("instructions")?.extract::<Vec<String>>()?;
        for inst_name in inst_map_instructions {
            // Prepare dictionary of instruction properties
            let mut out_prop: IndexMap<Option<HashableVec<u32>>, Option<InstructionProperties>> =
                IndexMap::new();
            let inst_map_qubit_instruction_for_name =
                inst_map.call_method1("qubits_with_instruction", (&inst_name,))?;
            let inst_map_qubit_instruction_for_name =
                inst_map_qubit_instruction_for_name.downcast::<PyList>()?;
            for qargs in inst_map_qubit_instruction_for_name {
                let qargs_: HashableVec<u32> =
                    if let Ok(qargs_to_tuple) = qargs.extract::<HashableVec<u32>>() {
                        qargs_to_tuple
                    } else {
                        HashableVec {
                            vec: vec![qargs.extract::<u32>()?],
                        }
                    };
                let mut props: Option<InstructionProperties> =
                    if let Some(Some(prop_value)) = self.gate_map.get(&inst_name) {
                        if let Some(prop) = prop_value.get(&Some(qargs_.clone())) {
                            prop.clone()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                let entry = get_calibration.call1((&inst_name, qargs_.to_owned()))?;
                let entry_comparison: bool = if let Some(props) = props.as_ref() {
                    !entry.eq(props._calibration.clone())?
                } else {
                    !entry.is_none()
                };
                if entry.getattr("user_provided")?.extract::<bool>()? && entry_comparison {
                    let mut duration: Option<f64> = None;
                    if let Some(dt) = self.dt {
                        if let Ok(entry_duration) =
                            entry.call_method0("get_schedule")?.getattr("duration")
                        {
                            duration = Some(dt * entry_duration.extract::<f64>()?);
                        }
                    }
                    props = Some(InstructionProperties::new(py, duration, None, Some(entry)));
                } else if props.is_none() {
                    continue;
                }

                if let Some(error_dict) = error_dict.as_ref() {
                    if let Some(error_dict_name) = error_dict.get(&inst_name) {
                        if let (Some(error_prop), Some(props_)) =
                            (error_dict_name.get(&qargs_), props.as_mut())
                        {
                            props_.error = Some(error_prop.to_owned().extract::<f64>()?);
                        }
                    }
                }
                out_prop.insert(Some(qargs_), props);
            }
            if out_prop.is_empty() {
                continue;
            }
            // Prepare Qiskit Gate object assigned to the entries
            if !self.gate_map.contains_key(&inst_name) {
                // Entry not found: Add new instruction
                if qiskit_inst_name_map.contains_key(&inst_name) {
                    // Remove qargs with length that doesn't match with instruction qubit number
                    let inst_obj = &qiskit_inst_name_map[&inst_name];
                    let mut normalized_props: IndexMap<
                        Option<HashableVec<u32>>,
                        Option<InstructionProperties>,
                    > = IndexMap::new();
                    for (qargs, prop) in out_prop.iter() {
                        if qargs
                            .as_ref()
                            .unwrap_or(&HashableVec { vec: vec![] })
                            .vec
                            .len()
                            != inst_obj.getattr("num_qubits")?.extract::<usize>()?
                        {
                            continue;
                        }
                        normalized_props.insert(qargs.to_owned(), prop.to_owned());
                    }
                    self.add_instruction(py, inst_obj, Some(normalized_props), Some(inst_name))?;
                } else {
                    // Check qubit length parameter name uniformity.
                    let mut qlen: HashSet<usize> = HashSet::new();
                    let mut param_names: HashSet<HashableVec<String>> = HashSet::new();
                    let inst_map_qubit_instruction_for_name =
                        inst_map.call_method1("qubits_with_instruction", (&inst_name,))?;
                    let inst_map_qubit_instruction_for_name =
                        inst_map_qubit_instruction_for_name.downcast::<PyList>()?;
                    for qargs in inst_map_qubit_instruction_for_name {
                        let qargs_ = if let Ok(qargs_ext) = qargs.extract::<HashableVec<u32>>() {
                            qargs_ext
                        } else {
                            HashableVec {
                                vec: vec![qargs.extract::<u32>()?],
                            }
                        };
                        qlen.insert(qargs_.vec.len());
                        let cal = if let Some(Some(prop)) = out_prop.get(&Some(qargs_.to_owned())) {
                            prop._calibration.as_ref()
                        } else {
                            None
                        };
                        if let Some(cal) = cal {
                            let params = cal
                                .call_method0(py, "get_signature")?
                                .getattr(py, "parameters")?
                                .call_method0(py, "keys")?
                                .extract::<HashableVec<String>>(py)?;
                            param_names.insert(params);
                        }
                        if qlen.len() > 1 || param_names.len() > 1 {
                            return Err(QiskitError::new_err(format!(
                                "Schedules for {:?} are defined non-uniformly for 
                            multiple qubit lengths {:?}, 
                            or different parameter names {:?}. 
                            Provide these schedules with inst_name_map or define them with 
                            different names for different gate parameters.",
                                &inst_name,
                                qlen.iter().collect::<Vec<_>>() as Vec<&usize>,
                                param_names.iter().collect::<Vec<_>>() as Vec<&HashableVec<String>>
                            )));
                        }
                        let gate_class = py.import_bound("qiskit.circuit.gate")?.getattr("Gate")?;
                        let parameter_class = py
                            .import_bound("qiskit.circuit.parameter")?
                            .getattr("Parameter")?;
                        let params =
                            parameter_class.call1((param_names.iter().next().cloned(),))?;
                        let kwargs = [
                            ("name", inst_name.to_owned().into_py(py)),
                            ("num_qubits", qlen.iter().next().to_owned().to_object(py)),
                            ("params", params.into_py(py)),
                        ]
                        .into_py_dict_bound(py);
                        let inst_obj = gate_class.call((), Some(&kwargs))?;
                        self.add_instruction(
                            py,
                            &inst_obj,
                            Some(out_prop.to_owned()),
                            Some(inst_name.to_owned()),
                        )?;
                    }
                }
            } else {
                // Entry found: Update "existing" instructions.
                for (qargs, prop) in out_prop.iter() {
                    if let Some(Some(gate_inst)) = self.gate_map.get(&inst_name) {
                        if !gate_inst.contains_key(qargs) {
                            continue;
                        }
                    }
                    self.update_instruction_properties(
                        py,
                        inst_name.to_owned(),
                        qargs.to_owned(),
                        prop.to_owned(),
                    )?;
                }
            }
        }
        Ok(())
    }

    #[pyo3(text_signature = "(/)")]
    fn instruction_schedule_map(
        &mut self,
        py: Python<'_>,
        out_inst_schedule_map: Bound<PyAny>,
    ) -> PyObject {
        /*
        Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions in the target with a pulse schedule defined.

        Returns:
            InstructionScheduleMap: The instruction schedule map for the
            instructions in this target with a pulse schedule defined.
         */
        if let Some(schedule_map) = self.instruction_schedule_map.clone() {
            return schedule_map;
        }
        for (instruction, qargs) in self.gate_map.iter() {
            if let Some(qargs) = qargs {
                for (qarg, properties) in qargs.iter() {
                    // Directly getting calibration entry to invoke .get_schedule().
                    // This keeps PulseQobjDef unparsed.
                    if let Some(properties) = properties {
                        let cal_entry = &properties._calibration;
                        if let Some(cal_entry) = cal_entry {
                            let _ = out_inst_schedule_map
                                .call_method1("_add", (instruction, qarg.clone(), cal_entry));
                        }
                    }
                }
            }
        }
        self.instruction_schedule_map = Some(out_inst_schedule_map.clone().unbind());
        out_inst_schedule_map.to_object(py)
    }

    #[pyo3(text_signature = "(/, operation)")]
    fn qargs_for_operation_name(
        &self,
        operation: String,
    ) -> PyResult<Option<Vec<Option<HashableVec<u32>>>>> {
        /*
        Get the qargs for a given operation name

        Args:
           operation (str): The operation name to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to.
         */
        if let Some(gate_map_oper) = self.gate_map.get(&operation).cloned() {
            if let Some(gate_map_op) = gate_map_oper {
                if gate_map_op.contains_key(&None) {
                    return Ok(None);
                }
                let qargs: Vec<Option<HashableVec<u32>>> = gate_map_op.into_keys().collect();
                Ok(Some(qargs))
            } else {
                Ok(None)
            }
        } else {
            Err(PyKeyError::new_err(format!(
                "Operation: {operation} not in Target."
            )))
        }
    }

    #[pyo3(text_signature = "(/,)")]
    fn durations(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.instruction_durations.is_some() {
            return Ok(self.instruction_durations.to_owned());
        }
        let mut out_durations: Vec<(&String, HashableVec<u32>, f64, &str)> = vec![];
        for (instruction, props_map) in self.gate_map.iter() {
            if let Some(props_map) = props_map {
                for (qarg, properties) in props_map.into_iter() {
                    if let Some(properties) = properties {
                        if let Some(duration) = properties.duration {
                            out_durations.push((
                                instruction,
                                qarg.to_owned().unwrap_or(HashableVec { vec: vec![] }),
                                duration,
                                "s",
                            ))
                        }
                    }
                }
            }
        }
        let instruction_duration_class = py
            .import_bound("qiskit.transpiler.instruction_durations")?
            .getattr("InstructionDurations")?;
        let kwargs = [("dt", self.dt)].into_py_dict_bound(py);
        self.instruction_durations = Some(
            instruction_duration_class
                .call((out_durations,), Some(&kwargs))?
                .unbind(),
        );
        Ok(self.instruction_durations.to_owned())
    }

    #[pyo3(text_signature = "(/,)")]
    fn timing_constraints(&self, py: Python<'_>) -> PyResult<PyObject> {
        let timing_constraints_class = py
            .import_bound("qiskit.transpiler.timing_constraints")?
            .getattr("TimingConstraints")?;
        Ok(timing_constraints_class
            .call1((
                self.granularity,
                self.min_length,
                self.pulse_alignment,
                self.acquire_alignment,
            ))?
            .unbind())
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn operations_for_qargs(
        &self,
        py: Python<'_>,
        qargs: Option<HashableVec<u32>>,
    ) -> PyResult<Py<PyList>> {
        let res = PyList::empty_bound(py);
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .vec
                .iter()
                .any(|x| !(0..(self.num_qubits.unwrap_or_default() as u32)).contains(x))
            {
                // TODO: Throw Python Exception
                return Err(PyKeyError::new_err(format!(
                    "{:?} not in target.",
                    qargs.vec
                )));
            }
        }
        if let Some(Some(gate_map_qarg)) = self.qarg_gate_map.get(&qargs) {
            for x in gate_map_qarg {
                res.append(self.gate_name_map[x].clone())?;
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(qarg) = self.global_operations.get(&qargs.vec.len()) {
                for arg in qarg {
                    res.append(arg)?;
                }
            }
        }
        for op in self.gate_name_map.values() {
            if isclass(py, op.bind(py))? {
                res.append(op)?;
            }
        }
        if res.is_empty() {
            return Err(PyKeyError::new_err(format!("{:?} not in target", {
                match &qargs {
                    Some(qarg) => format!("{:?}", qarg.vec),
                    None => "None".to_owned(),
                }
            })));
        }
        Ok(res.into())
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn operation_names_for_qargs(
        &self,
        py: Python<'_>,
        qargs: Option<HashableVec<u32>>,
    ) -> PyResult<HashSet<String>> {
        // When num_qubits == 0 we return globally defined operators
        let mut res = HashSet::new();
        let mut qargs = qargs;
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .vec
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default() as u32).contains(x))
            {
                return Err(PyKeyError::new_err(format!("{:?}", qargs)));
            }
        }
        if let Some(Some(qarg_gate_map_arg)) = self.qarg_gate_map.get(&qargs).as_ref() {
            res.extend(qarg_gate_map_arg.to_owned());
        }
        for (name, op) in self.gate_name_map.iter() {
            if isclass(py, op.bind(py))? {
                res.insert(name.into());
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(ext) = self.global_operations.get(&qargs.vec.len()) {
                res = ext.union(&res).cloned().collect();
            }
        }
        if res.is_empty() {
            return Err(PyKeyError::new_err(format!("{:?} not in target", qargs)));
        }
        Ok(res)
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn instruction_supported(
        &self,
        py: Python<'_>,
        parameter_class: &Bound<PyType>,
        check_obj_params: &Bound<PyAny>,
        operation_name: Option<String>,
        qargs: Option<HashableVec<u32>>,
        operation_class: Option<&Bound<PyAny>>,
        parameters: Option<&Bound<PyList>>,
    ) -> PyResult<bool> {
        // Fix num_qubits first, then think about this thing.
        let mut qargs = qargs;
        if self.num_qubits.is_none() {
            qargs = None;
        }

        if let Some(qargs_) = qargs.clone() {
            // For unique qarg comparisons
            let qarg_unique: HashSet<&u32> = HashSet::from_iter(qargs_.vec.iter());
            if let Some(operation_class) = operation_class {
                for (op_name, obj) in self.gate_name_map.iter() {
                    if isclass(py, obj.bind(py))? {
                        if !operation_class.eq(obj)? {
                            continue;
                        }
                        if qargs.is_none()
                            || (qargs_
                                .vec
                                .iter()
                                .all(|qarg| qarg <= &(self.num_qubits.unwrap_or_default() as u32))
                                && qarg_unique.len() == qargs_.vec.len())
                        {
                            return Ok(true);
                        } else {
                            return Ok(false);
                        }
                    }

                    if obj
                        .bind_borrowed(py)
                        .is_instance(operation_class.downcast::<PyType>()?)?
                    {
                        if let Some(parameters) = parameters {
                            if parameters.len()
                                != obj
                                    .getattr(py, "params")?
                                    .downcast_bound::<PyList>(py)?
                                    .len()
                            {
                                continue;
                            }
                            if !check_obj_params
                                .call1((parameters, obj))?
                                .extract::<bool>()?
                            {
                                continue;
                            }
                        }
                        if qargs.is_none() {
                            return Ok(true);
                        }
                        // TODO: Double check this method and what's stored in gate_map
                        if self.gate_map[op_name].is_none()
                            || self.gate_map[op_name].as_ref().unwrap().contains_key(&None)
                        {
                            let qubit_comparison = self.gate_name_map[op_name]
                                .getattr(py, "num_qubits")?
                                .extract::<usize>(py)?
                                == qargs_.vec.len()
                                && qargs_
                                    .vec
                                    .iter()
                                    .cloned()
                                    .all(|x| x < (self.num_qubits.unwrap_or_default() as u32));
                            return Ok(qubit_comparison);
                        }
                    }
                }
                return Ok(false);
            }
            if let Some(operation_name) = operation_name {
                if self.gate_map.contains_key(&operation_name) {
                    let mut obj = &self.gate_name_map[&operation_name];
                    if isclass(py, obj.bind(py))? {
                        // The parameters argument was set and the operation_name specified is
                        // defined as a globally supported class in the target. This means
                        // there is no available validation (including whether the specified
                        // operation supports parameters), the returned value will not factor
                        // in the argument `parameters`,

                        // If no qargs a operation class is supported
                        if qargs.is_none()
                            || (qargs_
                                .vec
                                .iter()
                                .all(|qarg| qarg <= &(self.num_qubits.unwrap_or_default() as u32))
                                && qarg_unique.len() == qargs_.vec.len())
                        {
                            return Ok(true);
                        } else {
                            return Ok(false);
                        }
                    }
                    let obj_params = obj.getattr(py, "params")?;
                    let obj_params = obj_params.downcast_bound::<PyList>(py)?;
                    if let Some(parameters) = parameters {
                        if parameters.len() != obj_params.len() {
                            return Ok(false);
                        }
                        for (index, param) in parameters.iter().enumerate() {
                            let mut matching_param = false;
                            if obj_params.get_item(index)?.is_instance(parameter_class)?
                                || param.eq(obj_params.get_item(index)?)?
                            {
                                matching_param = true;
                            }
                            if !matching_param {
                                return Ok(false);
                            }
                        }
                        return Ok(true);
                    }
                    if qargs.is_none() {
                        return Ok(true);
                    }
                    if let Some(gate_map_oper) = self.gate_map[&operation_name].as_ref() {
                        if gate_map_oper.contains_key(&qargs) {
                            return Ok(true);
                        }
                    }

                    // Double check this
                    if self.gate_map[&operation_name].is_none()
                        || self.gate_map[&operation_name]
                            .as_ref()
                            .unwrap()
                            .contains_key(&None)
                    {
                        obj = &self.gate_name_map[&operation_name];
                        if isclass(py, obj.bind(py))? {
                            if qargs.is_none()
                                || (qargs_.vec.iter().all(|qarg| {
                                    qarg <= &(self.num_qubits.unwrap_or_default() as u32)
                                }) && qarg_unique.len() == qargs_.vec.len())
                            {
                                return Ok(true);
                            } else {
                                return Ok(false);
                            }
                        } else {
                            let qubit_comparison = self.gate_name_map[&operation_name]
                                .getattr(py, "num_qubits")?
                                .extract::<usize>(py)?
                                == qargs_.vec.len()
                                && qargs_
                                    .vec
                                    .iter()
                                    .all(|x| x < &(self.num_qubits.unwrap_or_default() as u32));
                            return Ok(qubit_comparison);
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    #[pyo3(text_signature = "( /, operation_name: str, qargs: tuple[int, ...],)")]
    fn has_calibration(&self, operation_name: String, qargs: HashableVec<u32>) -> PyResult<bool> {
        /*
        Return whether the instruction (operation + qubits) defines a calibration.

        Args:
            operation_name: The name of the operation for the instruction.
            qargs: The tuple of qubit indices for the instruction.

        Returns:
            Returns ``True`` if the calibration is supported and ``False`` if it isn't.
         */
        if !self.gate_map.contains_key(&operation_name) {
            return Ok(false);
        }
        if let Some(gate_map_qarg) = self.gate_map[&operation_name].as_ref() {
            if let Some(oper_qarg) = &gate_map_qarg[&Some(qargs)] {
                return Ok(oper_qarg._calibration.is_some());
            } else {
                return Ok(false);
            }
        }
        Ok(false)
    }

    #[pyo3(text_signature = "( /, operation_name: str, qargs: tuple[int, ...],)")]
    fn get_calibration(
        &self,
        operation_name: String,
        qargs: HashableVec<u32>,
    ) -> PyResult<&PyObject> {
        /* Get calibrated pulse schedule for the instruction.

        If calibration is templated with parameters, one can also provide those values
        to build a schedule with assigned parameters.

        Args:
            operation_name: The name of the operation for the instruction.
            qargs: The tuple of qubit indices for the instruction.
            args: Parameter values to build schedule if any.
            kwargs: Parameter values with name to build schedule if any.

        Returns:
            Calibrated pulse schedule of corresponding instruction.
         */
        if !self.has_calibration(operation_name.clone(), qargs.clone())? {
            return Err(PyKeyError::new_err(format!(
                "Calibration of instruction {:?} for qubit {:?} is not defined.",
                operation_name, qargs.vec
            )));
        }

        Ok(
            self.gate_map[&operation_name].as_ref().unwrap()[&Some(qargs)]
                .as_ref()
                .unwrap()
                ._calibration
                .as_ref()
                .unwrap(),
        )
    }

    #[pyo3(text_signature = "(/, index: int)")]
    fn instruction_properties(&self, index: usize) -> PyResult<Option<InstructionProperties>> {
        let mut instruction_properties: Vec<Option<InstructionProperties>> = vec![];
        for operation in self.gate_map.keys() {
            if let Some(gate_map_oper) = self.gate_map[operation].to_owned() {
                for (_, inst_props) in gate_map_oper.iter() {
                    instruction_properties.push(inst_props.to_owned())
                }
            }
        }
        if !((0..instruction_properties.len()).contains(&index)) {
            return Err(PyIndexError::new_err(format!(
                "Index: {:?} is out of range.",
                index
            )));
        }
        Ok(instruction_properties[index].to_owned())
    }

    #[pyo3(text_signature = "(/, strict_direction=False)")]
    fn get_non_global_operation_names(&mut self, strict_direction: bool) -> PyResult<Vec<String>> {
        let mut search_set: HashSet<Option<HashableVec<u32>>> = HashSet::new();
        if strict_direction {
            if let Some(global_strict) = &self.non_global_strict_basis {
                return Ok(global_strict.to_owned());
            }
            // Build search set
            for qarg_key in self.qarg_gate_map.keys().cloned() {
                search_set.insert(qarg_key);
            }
        } else {
            if let Some(global_basis) = &self.non_global_basis {
                return Ok(global_basis.to_owned());
            }
            for qarg_key in self.qarg_gate_map.keys().cloned() {
                if let Some(qarg_key_) = &qarg_key {
                    if qarg_key_.vec.len() != 1 {
                        let mut vec = qarg_key_.clone().vec;
                        vec.sort();
                        let qarg_key = Some(HashableVec { vec });
                        search_set.insert(qarg_key);
                    }
                } else {
                    search_set.insert(qarg_key);
                }
            }
        }
        let mut incomplete_basis_gates: Vec<String> = vec![];
        let mut size_dict: IndexMap<usize, usize> = IndexMap::new();
        *size_dict
            .entry(1)
            .or_insert(self.num_qubits.unwrap_or_default()) = self.num_qubits.unwrap_or_default();
        for qarg in &search_set {
            if qarg.is_none()
                || qarg
                    .as_ref()
                    .unwrap_or(&HashableVec { vec: vec![] })
                    .vec
                    .len()
                    == 1
            {
                continue;
            }
            *size_dict
                .entry(
                    qarg.to_owned()
                        .unwrap_or(HashableVec { vec: vec![] })
                        .vec
                        .len(),
                )
                .or_insert(0) += 1;
        }
        for (inst, qargs) in self.gate_map.iter() {
            if let Some(qargs) = qargs {
                let mut qarg_len = qargs.len();
                let qarg_sample = qargs.keys().next();
                if let Some(qarg_sample) = qarg_sample {
                    if !strict_direction {
                        let mut qarg_set = HashSet::new();
                        for qarg in qargs.keys() {
                            let mut qarg_set_vec: HashableVec<u32> = HashableVec { vec: vec![] };
                            if let Some(qarg) = qarg {
                                let mut to_vec: Vec<u32> = qarg.vec.to_owned();
                                to_vec.sort();
                                qarg_set_vec = HashableVec { vec: to_vec };
                            }
                            qarg_set.insert(qarg_set_vec);
                        }
                        qarg_len = qarg_set.len();
                    }
                    if let Some(qarg_sample) = qarg_sample {
                        if qarg_len != *size_dict.entry(qarg_sample.vec.len()).or_insert(0) {
                            incomplete_basis_gates.push(inst.to_owned());
                        }
                    }
                }
            }
        }
        if strict_direction {
            self.non_global_strict_basis = Some(incomplete_basis_gates.to_owned());
            Ok(incomplete_basis_gates)
        } else {
            self.non_global_basis = Some(incomplete_basis_gates.to_owned());
            Ok(incomplete_basis_gates)
        }
    }

    // Class properties
    #[getter]
    fn qargs(&self) -> PyResult<Option<HashSet<Option<HashableVec<u32>>>>> {
        let qargs: HashSet<Option<HashableVec<u32>>> =
            self.qarg_gate_map.clone().into_keys().collect();
        // Modify logic to account for the case of {None}
        let next_entry = qargs.iter().flatten().next();
        if qargs.len() == 1 && (qargs.iter().next().is_none() || next_entry.is_none()) {
            return Ok(None);
        }
        Ok(Some(qargs))
    }

    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, PyObject)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, PyObject)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            if let Some(gate_map_op) = self.gate_map[op].as_ref() {
                for qarg in gate_map_op.keys() {
                    let instruction_pair =
                        (self.gate_name_map[op].clone(), qarg.clone().into_py(py));
                    instruction_list.push(instruction_pair);
                }
            }
        }
        // Return results.
        Ok(instruction_list)
    }

    #[getter]
    fn operation_names(&self) -> HashSet<String> {
        // Get the operation names in the target.
        return HashSet::from_iter(self.gate_map.keys().cloned());
    }

    #[getter]
    fn operations(&self) -> Vec<PyObject> {
        // Get the operation names in the target.
        return Vec::from_iter(self.gate_name_map.values().cloned());
    }

    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        // Returns a sorted list of physical qubits.
        Vec::from_iter(0..self.num_qubits.unwrap_or_default())
    }

    // Class methods
    #[classmethod]
    fn from_configuration(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        qubits_props_list_from_props: Bound<PyAny>,
        get_standard_gate_name_mapping: Bound<PyAny>,
        basis_gates: Vec<String>,
        num_qubits: Option<usize>,
        coupling_map: Option<PyObject>,
        inst_map: Option<Bound<PyAny>>,
        backend_properties: Option<&Bound<PyAny>>,
        instruction_durations: Option<PyObject>,
        concurrent_measurements: Option<Vec<Vec<usize>>>,
        dt: Option<f64>,
        timing_constraints: Option<PyObject>,
        custom_name_mapping: Option<IndexMap<String, Bound<PyAny>>>,
    ) -> PyResult<Self> {
        let mut num_qubits = num_qubits;
        let mut granularity: i32 = 1;
        let mut min_length: usize = 1;
        let mut pulse_alignment: i32 = 1;
        let mut acquire_alignment: i32 = 1;
        if let Some(timing_constraints) = timing_constraints {
            granularity = timing_constraints
                .getattr(py, "granularity")?
                .extract::<i32>(py)?;
            min_length = timing_constraints
                .getattr(py, "min_length")?
                .extract::<usize>(py)?;
            pulse_alignment = timing_constraints
                .getattr(py, "pulse_alignment")?
                .extract::<i32>(py)?;
            acquire_alignment = timing_constraints
                .getattr(py, "acquire_alignment")?
                .extract::<i32>(py)?;
        }
        let mut qubit_properties = None;
        if let Some(backend_properties) = backend_properties {
            let kwargs: Bound<PyDict> = [("properties", backend_properties)].into_py_dict_bound(py);
            qubit_properties = Some(
                qubits_props_list_from_props
                    .call((), Some(&kwargs))?
                    .extract::<Vec<PyObject>>()?,
            );
        }
        let mut target = Self::new(
            None,
            num_qubits,
            dt,
            Some(granularity),
            Some(min_length),
            Some(pulse_alignment),
            Some(acquire_alignment),
            qubit_properties,
            concurrent_measurements,
        );
        let name_mapping = get_standard_gate_name_mapping.call0()?;
        let name_mapping = name_mapping.downcast::<PyDict>()?;
        if let Some(custom_name_mapping) = custom_name_mapping {
            name_mapping.call_method1("update", (custom_name_mapping,))?;
        }

        /*
           While BackendProperties can also contain coupling information we
           rely solely on CouplingMap to determine connectivity. This is because
           in legacy transpiler usage (and implicitly in the BackendV1 data model)
           the coupling map is used to define connectivity constraints and
           the properties is only used for error rate and duration population.
           If coupling map is not specified we ignore the backend_properties
        */
        if let Some(coupling_map) = coupling_map {
            let mut one_qubit_gates: Vec<String> = vec![];
            let mut two_qubit_gates: Vec<String> = vec![];
            let mut global_ideal_variable_width_gates: Vec<String> = vec![];
            if num_qubits.is_none() {
                num_qubits = Some(
                    coupling_map
                        .getattr(py, "graph")?
                        .call_method0(py, "edge_list")?
                        .downcast_bound::<PyList>(py)?
                        .len(),
                )
            }
            for gate in basis_gates {
                if let Some(gate_obj) = name_mapping.get_item(&gate)? {
                    let gate_obj_num_qubits = gate_obj.getattr("num_qubits")?.extract::<usize>()?;
                    if gate_obj_num_qubits == 1 {
                        one_qubit_gates.push(gate);
                    } else if gate_obj_num_qubits == 2 {
                        two_qubit_gates.push(gate);
                    } else if isclass(py, &gate_obj)? {
                        global_ideal_variable_width_gates.push(gate)
                    } else {
                        return Err(TranspilerError::new_err(
                            format!(
                                "The specified basis gate: {gate} has {gate_obj_num_qubits} 
                                qubits. This constructor method only supports fixed width operations 
                                with <= 2 qubits (because connectivity is defined on a CouplingMap)."
                            )
                        ));
                    }
                } else {
                    return Err(PyKeyError::new_err(format!(
                        "The specified basis gate: {gate} is not present in the standard gate
                            names or a provided custom_name_mapping"
                    )));
                }
            }
            for gate in one_qubit_gates {
                let mut gate_properties: IndexMap<
                    Option<HashableVec<u32>>,
                    Option<InstructionProperties>,
                > = IndexMap::new();
                for qubit in 0..num_qubits.unwrap_or_default() {
                    let mut error: Option<f64> = None;
                    let mut duration: Option<f64> = None;
                    let mut calibration: Option<Bound<PyAny>> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, qubit))?
                                .extract::<f64>()
                            {
                                Ok(duration) => Some(duration),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties
                            .call_method1("gate_error", (&gate, qubit))?
                            .extract::<f64>()
                        {
                            Ok(error) => Some(error),
                            Err(_) => None,
                        };
                    }
                    if let Some(inst_map) = &inst_map {
                        calibration = match inst_map
                            .call_method1("_get_calibration_entry", (&gate, qubit))
                        {
                            Ok(calibration) => {
                                if dt.is_some()
                                    && calibration.getattr("user_provided")?.extract::<bool>()?
                                {
                                    duration = Some(
                                        calibration
                                            .call_method0("get_schedule")?
                                            .getattr("duration")?
                                            .extract::<f64>()?
                                            * dt.unwrap_or_default(),
                                    );
                                }
                                Some(calibration)
                            }
                            Err(_) => None,
                        }
                    }
                    if let Some(instruction_durations) = &instruction_durations {
                        let kwargs = [("unit", "s")].into_py_dict_bound(py);
                        duration = match instruction_durations
                            .call_method_bound(py, "get", (&gate, qubit), Some(&kwargs))?
                            .extract::<f64>(py)
                        {
                            Ok(duration) => Some(duration),
                            Err(_) => None,
                        }
                    }
                    if error.is_none() && duration.is_none() && calibration.is_none() {
                        gate_properties.insert(
                            Some(HashableVec {
                                vec: vec![qubit as u32],
                            }),
                            None,
                        );
                    } else {
                        gate_properties.insert(
                            Some(HashableVec {
                                vec: vec![qubit as u32],
                            }),
                            Some(InstructionProperties::new(py, duration, error, calibration)),
                        );
                    }
                }
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(py, &inst, Some(gate_properties), Some(gate))?;
                }
            }
            let edges = coupling_map
                .call_method0(py, "get_edges")?
                .extract::<Vec<[u32; 2]>>(py)?;
            for gate in two_qubit_gates {
                let mut gate_properties: IndexMap<
                    Option<HashableVec<u32>>,
                    Option<InstructionProperties>,
                > = IndexMap::new();
                for edge in edges.as_slice().iter().copied() {
                    let mut error: Option<f64> = None;
                    let mut duration: Option<f64> = None;
                    let mut calibration: Option<Bound<PyAny>> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, edge))?
                                .extract::<f64>()
                            {
                                Ok(duration) => Some(duration),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties
                            .call_method1("gate_error", (&gate, edge))?
                            .extract::<f64>()
                        {
                            Ok(error) => Some(error),
                            Err(_) => None,
                        };
                    }
                    if let Some(inst_map) = &inst_map {
                        calibration = match inst_map
                            .call_method1("_get_calibration_entry", (&gate, edge))
                        {
                            Ok(calibration) => {
                                if dt.is_some()
                                    && calibration.getattr("user_provided")?.extract::<bool>()?
                                {
                                    duration = Some(
                                        calibration
                                            .call_method0("get_schedule")?
                                            .getattr("duration")?
                                            .extract::<f64>()?
                                            * dt.unwrap_or_default(),
                                    );
                                }
                                Some(calibration)
                            }
                            Err(_) => None,
                        }
                    }
                    if let Some(instruction_durations) = &instruction_durations {
                        let kwargs = [("unit", "s")].into_py_dict_bound(py);
                        duration = match instruction_durations
                            .call_method_bound(py, "get", (&gate, edge), Some(&kwargs))?
                            .extract::<f64>(py)
                        {
                            Ok(duration) => Some(duration),
                            Err(_) => None,
                        }
                    }
                    if error.is_none() && duration.is_none() && calibration.is_none() {
                        gate_properties.insert(
                            Some(HashableVec {
                                vec: edge.into_iter().collect(),
                            }),
                            None,
                        );
                    } else {
                        gate_properties.insert(
                            Some(HashableVec {
                                vec: edge.into_iter().collect(),
                            }),
                            Some(InstructionProperties::new(py, duration, error, calibration)),
                        );
                    }
                }
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(py, &inst, Some(gate_properties), Some(gate))?;
                }
            }
            for gate in global_ideal_variable_width_gates {
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(py, &inst, None, Some(gate))?;
                }
            }
        } else {
            for gate in basis_gates {
                if !name_mapping.contains(&gate)? {
                    return Err(PyKeyError::new_err(format!(
                        "The specified basis gate: {gate} is not present in the standard gate
                            names or a provided custom_name_mapping"
                    )));
                }
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(py, &inst, None, Some(gate))?;
                }
            }
        }
        Ok(target)
    }

    // Magic methods:
    fn __iter__(&self) -> PyResult<Vec<String>> {
        Ok(self.gate_map.keys().cloned().collect())
    }

    fn __getitem__(&self, key: String) -> PyResult<TargetValue> {
        if let Some(value) = self.gate_map.get(&key) {
            Ok(value.to_owned())
        } else {
            Err(PyKeyError::new_err(format!("{key} not in gate_map")))
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __contains__(&self, item: String) -> PyResult<bool> {
        Ok(self.gate_map.contains_key(&item))
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self.gate_map.keys().cloned().collect())
    }

    fn values(&self) -> PyResult<Vec<TargetValue>> {
        Ok(self.gate_map.values().cloned().collect())
    }

    fn items(&self) -> PyResult<Vec<(String, TargetValue)>> {
        Ok(self.gate_map.clone().into_iter().collect())
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
