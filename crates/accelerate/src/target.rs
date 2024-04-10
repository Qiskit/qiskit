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

use hashbrown::{HashMap, HashSet};
use pyo3::{
    prelude::*,
    pyclass,
    types::{PyDict, PyList, PyTuple},
};

// TEMPORARY: until I can wrap around Python class
pub fn is_instance(py: Python<'_>, object: &PyObject, class_names: HashSet<String>) -> bool {
    // Get type name
    let type_name: Option<String> = match object.getattr(py, "__class__").ok() {
        Some(class) => class
            .getattr(py, "__name__")
            .ok()
            .map(|name| name.extract::<String>(py).ok().unwrap_or("".to_string())),
        None => None,
    };
    // Check if it matches any option
    match type_name {
        Some(class_name) => class_names.contains(&class_name),
        None => false,
    }
}

// TEMPORARY: until I can wrap python class.
pub fn is_class(py: Python<'_>, object: &PyObject) -> bool {
    // If item is a Class, it must have __name__ property.
    object.getattr(py, "__name__").is_ok()
}

// TEMPORARY: Helper function to import class or method from function.
pub fn import_from_module<'a>(py: Python<'a>, module: &str, method: &str) -> Bound<'a, PyAny> {
    match py.import_bound(module) {
        Ok(py_mod) => match py_mod.getattr(method) {
            Ok(obj) => obj,
            Err(e) => panic!(
                "Could not find '{:?} in module '{:?}': {:?}.",
                method.to_string(),
                module.to_string(),
                e
            ),
        },
        Err(e) => panic!("Could not find module '{:?}': {:?}", &module, e),
    }
}

// TEMPORARY: Helper function to import class or method from function.
pub fn import_from_module_call1<'a>(
    py: Python<'a>,
    module: &str,
    method: &str,
    args: impl IntoPy<Py<PyTuple>>,
) -> Bound<'a, PyAny> {
    let result = import_from_module(py, module, method);
    match result.call1(args) {
        Ok(res) => res,
        Err(e) => panic!("Could not call on method '{:?}': {:?}", method, e),
    }
}

pub fn import_from_module_call0<'a>(
    py: Python<'a>,
    module: &str,
    method: &str,
) -> Bound<'a, PyAny> {
    let result = import_from_module(py, module, method);
    match result.call0() {
        Ok(res) => res,
        Err(e) => panic!("Could not call on method '{:?}': {:?}", method, e),
    }
}

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

#[pyclass(module = "qiskit._accelerate.target.InstructionProperties")]
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f32>,
    #[pyo3(get)]
    pub error: Option<f32>,
    pub calibration: Option<PyObject>,
    calibration_: Option<PyObject>,
}

#[pymethods]
impl InstructionProperties {
    #[new]
    #[pyo3(text_signature = "(/, duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,)")]
    pub fn new(duration: Option<f32>, error: Option<f32>, calibration: Option<PyObject>) -> Self {
        InstructionProperties {
            calibration,
            error,
            duration,
            calibration_: Option::<PyObject>::None,
        }
    }

    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> Option<PyObject> {
        match &self.calibration_ {
            Some(calibration) => calibration.call_method0(py, "get_schedule").ok(),
            None => None,
        }
    }

    #[setter]
    pub fn set_calibration(&mut self, py: Python<'_>, calibration: PyObject) {
        // Conditional new entry
        let new_entry = if is_instance(
            py,
            &calibration,
            HashSet::from(["Schedule".to_string(), "ScheduleBlock".to_string()]),
        ) {
            // TEMPORARY: Import calibration_entries module
            let module = match py.import_bound("qiskit.pulse.calibration_entries") {
                Ok(module) => module,
                Err(e) => panic!(
                    "Could not find the module qiskit.pulse.calibration_entries: {:?}",
                    e
                ),
            };
            // TEMPORARY: Import SchedDef class object
            let sched_def = match module.call_method0("ScheduleDef") {
                Ok(sched) => sched.to_object(py),
                Err(e) => panic!("Failed to import the 'ScheduleDef' class: {:?}", e),
            };

            // TEMPORARY: Send arguments for the define call.
            let args = (&calibration, true);
            // Peform the function call.
            sched_def.call_method1(py, "define", args).ok();
            sched_def
        } else {
            calibration
        };
        self.calibration_ = Some(new_entry);
    }
}

#[pyclass(mapping, module = "qiskit._accelerate.target.Target")]
#[derive(Clone, Debug)]
pub struct Target {
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get)]
    pub num_qubits: usize,
    #[pyo3(get)]
    pub dt: Option<f32>,
    #[pyo3(get)]
    pub granularity: i32,
    #[pyo3(get)]
    pub min_length: usize,
    #[pyo3(get)]
    pub pulse_alignment: i32,
    #[pyo3(get)]
    pub acquire_alignment: i32,
    #[pyo3(get)]
    pub qubit_properties: Vec<PyObject>,
    #[pyo3(get)]
    pub concurrent_measurements: Vec<HashSet<usize>>,
    // Maybe convert PyObjects into rust representations of Instruction and Data
    gate_map: HashMap<String, HashMap<HashableVec<u32>, PyObject>>,
    gate_name_map: HashMap<String, PyObject>,
    global_operations: HashMap<usize, HashSet<String>>,
    qarg_gate_map: HashMap<HashableVec<u32>, HashSet<String>>,
    instructions_durations: Option<PyObject>,
    instruction_schedule_map: Option<PyObject>,
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
        dt: Option<f32>,
        granularity: Option<i32>,
        min_length: Option<usize>,
        pulse_alignment: Option<i32>,
        acquire_alignment: Option<i32>,
        qubit_properties: Option<Vec<PyObject>>,
        concurrent_measurements: Option<Vec<HashSet<usize>>>,
    ) -> Self {
        Target {
            description: description.unwrap_or("".to_string()),
            num_qubits: num_qubits.unwrap_or(0),
            dt,
            granularity: granularity.unwrap_or(1),
            min_length: min_length.unwrap_or(1),
            pulse_alignment: pulse_alignment.unwrap_or(1),
            acquire_alignment: acquire_alignment.unwrap_or(0),
            qubit_properties: qubit_properties.unwrap_or(Vec::new()),
            concurrent_measurements: concurrent_measurements.unwrap_or(Vec::new()),
            gate_map: HashMap::new(),
            gate_name_map: HashMap::new(),
            global_operations: HashMap::new(),
            qarg_gate_map: HashMap::new(),
            instructions_durations: Option::None,
            instruction_schedule_map: Option::None,
        }
    }

    #[pyo3(text_signature = "(/, instruction, properties=None, name=None")]
    fn add_instruction(
        &mut self,
        py: Python<'_>,
        instruction: PyObject,
        properties: Option<Bound<PyDict>>,
        name: Option<String>,
    ) {
        // Unwrap instruction name
        let mut instruction_name: String = match name {
            Some(name) => name,
            None => "".to_string(),
        };
        // Unwrap instruction num qubits
        let instruction_num_qubits = match instruction.getattr(py, "num_qubits") {
            Ok(number) => match number.extract::<usize>(py) {
                Ok(num_qubits) => num_qubits,
                Err(e) => panic!(
                    "The provided instruction does not have a valid number of qubits: {:?}",
                    e
                ),
            },
            Err(e) => panic!(
                "The provided instruction does not the attribute 'num_qubits': {:?}",
                e
            ),
        };
        let is_class: bool = is_class(py, &instruction);
        // Cont. unwrap instruction name
        if !is_class {
            if instruction_name.is_empty() {
                instruction_name = match instruction.getattr(py, "name") {
                    Ok(i_name) => match i_name.extract::<String>(py) {
                        Ok(i_name) => i_name,
                        Err(e) => panic!("The provided instruction does not have a valid 'name' attribute: {:?}", e)
                    },
                    Err(e) => panic!("A name must be specified when defining a supported global operation by class: {:?}", e)
                };
            }
        } else {
            if instruction_name.is_empty() {
                panic!(
                    "A name must be specified when defining a supported global operation by class."
                );
            }
            if properties.is_none() {
                panic!("An instruction added globally by class can't have properties set.");
            }
        }

        // Unwrap properties
        let properties: Bound<PyDict> = properties.unwrap_or(PyDict::new_bound(py));
        // Check if instruction exists
        if self.gate_map.contains_key(&instruction_name) {
            panic!(
                "Instruction {:?} is already in the target",
                &instruction_name
            );
        }
        // Add to gate name map
        self.gate_name_map
            .insert(instruction_name.clone(), instruction);

        // TEMPORARY: Build qargs with hashed qargs.
        let mut qargs_val: HashMap<HashableVec<u32>, PyObject> = HashMap::new();
        if !is_class {
            // If no properties
            if properties.is_empty() {
                if let Some(operation) = self.global_operations.get_mut(&instruction_num_qubits) {
                    operation.insert(instruction_name.clone());
                } else {
                    self.global_operations.insert(
                        instruction_num_qubits,
                        HashSet::from([instruction_name.clone()]),
                    );
                }
            }
            // Obtain nested qarg hashmap
            for (qarg, values) in properties {
                // Obtain values of qargs
                let qarg = match qarg.extract::<Vec<u32>>().ok() {
                    Some(vec) => HashableVec { vec },
                    None => HashableVec { vec: vec![] },
                };
                // Store qargs hash value.
                // self.qarg_hash_table.insert(qarg_hash, qarg.clone());
                if !qarg.vec.is_empty() && qarg.vec.len() != instruction_num_qubits {
                    panic!("The number of qubits for {:?} does not match the number of qubits in the properties dictionary: {:?}", &instruction_name, qarg.vec)
                }
                if !qarg.vec.is_empty() {
                    self.num_qubits = self
                        .num_qubits
                        .max(qarg.vec.iter().cloned().fold(0, u32::max) as usize + 1)
                }
                qargs_val.insert(qarg.clone(), values.to_object(py));
                if let Some(gate_map_key) = self.qarg_gate_map.get_mut(&qarg) {
                    gate_map_key.insert(instruction_name.clone());
                } else {
                    self.qarg_gate_map
                        .insert(qarg, HashSet::from([instruction_name.clone()]));
                }
            }
        }
        self.gate_map.insert(instruction_name, qargs_val);
    }

    #[pyo3(text_signature = "(/, instruction, qargs, properties)")]
    fn update_instruction_properties(
        &mut self,
        _py: Python<'_>,
        instruction: String,
        qargs: Vec<u32>,
        properties: PyObject,
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
            panic!(
                "Provided instruction : '{:?}' not in this Target.",
                &instruction
            );
        };
        let qargs = HashableVec { vec: qargs };
        if !self.gate_map[&instruction].contains_key(&qargs) {
            panic!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs, &instruction
            );
        }
        if let Some(q_vals) = self.gate_map.get_mut(&instruction) {
            if let Some(qvals_qargs) = q_vals.get_mut(&qargs) {
                *qvals_qargs = properties
            }
        }
        self.instructions_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
    }

    #[getter]
    fn instructions(&self) -> PyResult<Vec<(PyObject, Vec<u32>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Vec<u32>)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            for qarg in self.gate_map[op].keys() {
                let instruction_pair = (self.gate_name_map[op].clone(), qarg.vec.clone());
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }

    #[pyo3(text_signature = "(/, inst_map, inst_name_map=None, error_dict=None")]
    fn update_from_instruction_schedule_map(
        &mut self,
        py: Python<'_>,
        inst_map: Bound<PyAny>,
        inst_name_map: Option<HashMap<String, Bound<PyAny>>>,
        error_dict: Option<Bound<PyDict>>,
    ) -> PyResult<()> {
        println!("I'm inside but...??");
        let get_calibration = inst_map.getattr("_get_calibration_entry")?;

        // Expand name mapping with custom gate namer provided by user.
        // TEMPORARY: Get arround improting this module and function, will be fixed after python wrapping.
        let qiskit_inst_name_map = import_from_module_call0(
            py,
            "qiskit.circuit.library.standard_gates",
            "get_standard_gate_name_mapping",
        );

        // Update with inst_name_map if possible.
        if inst_name_map.is_some() {
            qiskit_inst_name_map.call_method1("update", (inst_name_map,))?;
        }

        for inst_name in inst_map.getattr("instructions")?.extract::<Vec<String>>()? {
            let mut out_props: HashMap<HashableVec<u32>, PyObject> = HashMap::new();
            for qargs in inst_map
                .call_method1("qubits_with_instruction", (&inst_name,))?
                .downcast::<PyList>()?
            {
                let qargs = HashableVec {
                    vec: if let Ok(qargs) = qargs.extract::<u32>() {
                        vec![qargs]
                    } else {
                        qargs.extract::<Vec<u32>>()?
                    },
                };
                let mut props: Option<PyObject> = if self.gate_map[&inst_name].contains_key(&qargs)
                {
                    Some(self.gate_map[&inst_name][&qargs].clone())
                } else {
                    None
                };
                let entry = get_calibration.call1((&inst_name, qargs.vec.clone()))?;
                let prop_entry = if let Some(prop) = &props {
                    entry.eq(prop.getattr(py, "_calibration")?)?
                } else {
                    entry.is_none()
                };
                if entry.getattr("user_provided")?.extract::<bool>()? && prop_entry {
                    // It only copies user-provided calibration from the inst map.
                    // Backend defined entry must already exist in Target.
                    let duration = if let Some(dur) = self.dt {
                        Some(
                            entry
                                .call_method0("get_schedule")?
                                .getattr("duration")?
                                .extract::<f32>()?
                                * dur,
                        )
                    } else {
                        None
                    };
                    props = Some(
                        import_from_module_call1(
                            py,
                            "qiskit.transpiler.target",
                            "Target",
                            (duration, None::<f32>, entry.clone()),
                        )
                        .into(),
                    );
                } else {
                    if props.is_none() {
                        // Edge case. Calibration is backend defined, but this is not
                        // registered in the backend target. Ignore this entry.
                        continue;
                    }
                }
                if let (Some(error_dict), Some(props)) = (error_dict.clone(), props.clone()) {
                    if let Some(inst_dict) = error_dict.get_item(inst_name.clone())? {
                        props.setattr(
                            py,
                            "error",
                            inst_dict.get_item(PyTuple::new_bound(py, qargs.vec.clone()))?,
                        )?;
                    }
                    if let Some(out_p) = out_props.get_mut(&qargs) {
                        *out_p = props.clone();
                    }
                }
                if let (Some(out_p), Some(props)) = (out_props.get_mut(&qargs), props) {
                    *out_p = props.to_object(py);
                }
                if out_props.is_empty() {
                    continue;
                }
            }
            // Prepare Qiskit Gate object assigned to the entries
            if !self.gate_map.contains_key(&inst_name) {
                if qiskit_inst_name_map.contains(&inst_name)? {
                    let inst_obj = qiskit_inst_name_map.get_item(&inst_name)?;
                    let normalized_props = PyDict::new_bound(py);
                    for (qargs, prop) in out_props.iter() {
                        if qargs.vec.len() != inst_obj.getattr("num_qubits")?.extract::<usize>()? {
                            continue;
                        }
                        normalized_props
                            .set_item(PyTuple::new_bound(py, qargs.vec.clone()), prop)?;
                    }
                    self.add_instruction(
                        py,
                        inst_obj.to_object(py),
                        Some(normalized_props),
                        Some(inst_name.clone()),
                    );
                } else {
                    // Check qubit length parameter name uniformity.
                    let mut qlen: HashSet<usize> = HashSet::new();
                    let mut param_names: HashSet<HashableVec<String>> = HashSet::new();
                    for qarg in inst_map
                        .call_method1("qubit_with_instruction", (&inst_name,))?
                        .downcast::<PyList>()?
                    {
                        let extract_qarg: HashableVec<u32>;
                        if is_instance(py, &qarg.to_object(py), HashSet::from(["int".to_string()]))
                        {
                            extract_qarg = HashableVec {
                                vec: vec![qarg.extract::<u32>()?],
                            };
                        } else {
                            extract_qarg = HashableVec {
                                vec: qarg.extract::<Vec<u32>>()?,
                            };
                        }
                        qlen.insert(extract_qarg.vec.len());
                        let cal = out_props[&extract_qarg].getattr(py, "_calibration")?;
                        param_names.insert(HashableVec {
                            vec: cal
                                .call_method0(py, "get_signature")?
                                .getattr(py, "parameters")?
                                .call_method0(py, "keys")?
                                .extract::<Vec<String>>(py)?,
                        });
                    }
                    if qlen.len() > 1 && param_names.len() > 1 {
                        panic!(
                            "Schedules for {:?} are defined non-uniformly for 
                        multiple qubit lengths {:?}, 
                        or different parameter names {:?}. 
                        Provide these schedules with inst_name_map or define them with 
                        different names for different gate parameters.",
                            inst_name, qlen, param_names
                        )
                    }
                    let params_list = if let Some(params) = param_names
                        .into_iter()
                        .next()
                        .and_then(|hashvec| Some(hashvec.vec))
                    {
                        let mut param_vec = vec![];
                        let _ = params.into_iter().map(|name| {
                            param_vec.push(import_from_module_call1(
                                py,
                                "qiskit.circuit.parameters",
                                "Parameter",
                                (&name,),
                            ))
                        });
                        param_vec
                    } else {
                        vec![]
                    };

                    let inst_obj = import_from_module_call1(
                        py,
                        "qiskit.circuit.gate",
                        "Gate",
                        (&inst_name, qlen.into_iter().next(), params_list),
                    );
                    let parsed_dict = PyDict::new_bound(py);
                    let _ = out_props.iter().map(|(key, val)| {
                        parsed_dict.set_item(PyTuple::new_bound(py, key.vec.clone()), val)
                    });
                    self.add_instruction(
                        py,
                        inst_obj.to_object(py),
                        Some(parsed_dict),
                        Some(inst_name.clone()),
                    )
                }
            } else {
                for (qargs, prop) in out_props.into_iter() {
                    if self.gate_map[&inst_name].contains_key(&qargs) {
                        continue;
                    }
                    return self.update_instruction_properties(py, inst_name.clone(), qargs.vec, prop);
                }
            }
        }
        Ok(())
    }

    #[pyo3(text_signature = "/")]
    fn instruction_schedule_map(&mut self, py: Python<'_>) -> Option<PyObject> {
        /*
        Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions in the target with a pulse schedule defined.

        Returns:
            InstructionScheduleMap: The instruction schedule map for the
            instructions in this target with a pulse schedule defined.
         */
        if self.instruction_schedule_map.is_some() {
            return self.instruction_schedule_map.clone();
        }
        let out_inst_schedule_map = import_from_module_call0(
            py,
            "qiskit.pulse.instruction_schedule_map",
            "InstructionScheduleMap",
        )
        .to_object(py);
        for (instruction, qargs) in self.gate_map.iter() {
            for (qarg, properties) in qargs.iter() {
                // Directly getting calibration entry to invoke .get_schedule().
                // This keeps PulseQobjDef unparsed.
                let cal_entry = properties.getattr(py, "_calibration").ok();
                if let Some(cal_entry) = cal_entry {
                    let _ = out_inst_schedule_map.call_method1(
                        py,
                        "_add",
                        (instruction, qarg.vec.clone(), cal_entry),
                    );
                }
            }
        }
        self.instruction_schedule_map = Some(out_inst_schedule_map.clone());
        Some(out_inst_schedule_map)
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
