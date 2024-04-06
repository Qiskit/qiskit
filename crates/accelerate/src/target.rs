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

use hashbrown::{HashMap, HashSet};
use pyo3::{prelude::*, pyclass, types::PyDict};

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

pub fn is_class(py: Python<'_>, object: &PyObject) -> bool {
    // If item is a Class, it must have __name__ property.
    object.getattr(py, "__name__").is_ok()
}

#[pyclass(module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
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
            // Import calibration_entries module
            let module = match py.import_bound("qiskit.pulse.calibration_entries") {
                Ok(module) => module,
                Err(e) => panic!(
                    "Could not find the module qiskit.pulse.calibration_entries: {:?}",
                    e
                ),
            };
            // Import SchedDef class object
            let sched_def = match module.call_method0("ScheduleDef") {
                Ok(sched) => sched.to_object(py),
                Err(e) => panic!("Failed to import the 'ScheduleDef' class: {:?}", e),
            };

            // Send arguments for the define call.
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

#[pyclass(mapping, module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
pub struct Target {
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get)]
    pub num_qubits: usize,
    #[pyo3(get)]
    pub dt: f32,
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
    gate_map: HashMap<String, HashMap<isize, PyObject>>,
    gate_name_map: HashMap<String, PyObject>,
    global_operations: HashMap<usize, HashSet<String>>,
    qarg_gate_map: HashMap<isize, HashSet<String>>,
    qarg_hash_table: HashMap<isize, Vec<usize>>,
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
            dt: dt.unwrap_or(0.0),
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
            qarg_hash_table: HashMap::new(),
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
        let mut qargs_val: HashMap<isize, PyObject> = HashMap::new();
        if !is_class {
            // If no properties
            if properties.is_empty() {
                if self.global_operations.contains_key(&instruction_num_qubits) {
                    self.global_operations
                        .get_mut(&instruction_num_qubits)
                        .unwrap()
                        .insert(instruction_name.clone());
                } else {
                    self.global_operations.insert(
                        instruction_num_qubits,
                        HashSet::from([instruction_name.clone()]),
                    );
                }
            }

            // Obtain nested qarg hashmap
            for (qarg, values) in properties {
                // Obtain qarg hash for mapping
                let qarg_hash = match qarg.hash() {
                    Ok(vec) => vec,
                    Err(e) => panic!(
                        "Failed to hash q_args from the provided properties: {:?}.",
                        e
                    ),
                };
                // Obtain values of qargs
                let qarg = match qarg.extract::<Vec<usize>>().ok() {
                    Some(vec) => vec,
                    None => vec![],
                };
                // Store qargs hash value.
                self.qarg_hash_table.insert(qarg_hash, qarg.clone());
                if !qarg.is_empty() && qarg.len() != instruction_num_qubits {
                    panic!("The number of qubits for {:?} does not match the number of qubits in the properties dictionary: {:?}", &instruction_name, qarg)
                }
                if !qarg.is_empty() {
                    self.num_qubits = self
                        .num_qubits
                        .max(qarg.iter().cloned().fold(0, usize::max) + 1)
                }
                qargs_val.insert(qarg_hash, values.to_object(py));
                if self.qarg_gate_map.contains_key(&qarg_hash) {
                    self.qarg_gate_map
                        .get_mut(&qarg_hash)
                        .unwrap()
                        .insert(instruction_name.clone());
                } else {
                    self.qarg_gate_map
                        .insert(qarg_hash, HashSet::from([instruction_name.clone()]));
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
        qargs: Bound<PyAny>,
        properties: PyObject,
    ) {
        /* Update the property object for an instruction qarg pair already in the Target

        Args:
            instruction (str): The instruction name to update
            qargs (tuple): The qargs to update the properties of
            properties (InstructionProperties): The properties to set for this instruction
        Raises:
            KeyError: If ``instruction`` or ``qarg`` are not in the target */
        
        // For debugging
        // println!(
        //     "Before - {:?}: {:?}",
        //     instruction, self.gate_map[&instruction]
        // );
        if !self.gate_map.contains_key(&instruction) {
            panic!(
                "Provided instruction : '{:?}' not in this Target.",
                &instruction
            );
        };
        if !self.gate_map[&instruction].contains_key(&qargs.hash().ok().unwrap()) {
            panic!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs, &instruction
            );
        }
        self.gate_map.get_mut(&instruction).map(|q_vals| {
            *q_vals.get_mut(&qargs.hash().ok().unwrap()).unwrap() = properties;
            Some(())
        });
        self.instructions_durations = Option::None;
        self.instruction_schedule_map = Option::None;
        // println!(
        //     "After - {:?}: {:?}",
        //     instruction, self.gate_map[&instruction]
        // );
    }

    #[getter]
    fn instructions(&self) -> PyResult<Vec<(PyObject, Vec<usize>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Vec<usize>)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            for qarg in self.gate_map[op].keys() {
                let instruction_pair = (
                    self.gate_name_map[op].clone(),
                    self.qarg_hash_table[qarg].clone(),
                );
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
