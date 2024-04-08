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

use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
};

use hashbrown::{HashMap, HashSet};
use pyo3::{prelude::*, pyclass, types::PyDict};

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

#[derive(Eq, PartialEq, Clone, Debug)]
struct Qargs {
    pub qargs: Vec<u32>,
}

impl Hash for Qargs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for qarg in self.qargs.iter() {
            qarg.hash(state);
        }
    }
}

// TEMPORARY: Helper function to import class or method from function.
pub fn import_from_module_call<'a>(
    py: Python<'a>,
    module: &str,
    method: &str,
    args: Option<()>,
) -> Bound<'a, PyAny> {
    let result = import_from_module(py, module, method);
    match args {
        Some(arg) => match result.call1(arg) {
            Ok(res) => res,
            Err(e) => panic!("Could not call on method '{:?}': {:?}", method, e),
        },
        None => match result.call0() {
            Ok(res) => res,
            Err(e) => panic!("Could not call on method '{:?}': {:?}", method, e),
        },
    }
}

#[pyclass(module = "qiskit._accelerate.target.InstructionProperties")]
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
    gate_map: HashMap<String, HashMap<Qargs, PyObject>>,
    gate_name_map: HashMap<String, PyObject>,
    global_operations: HashMap<usize, HashSet<String>>,
    qarg_gate_map: HashMap<Qargs, HashSet<String>>,
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
            dt: dt,
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
        let mut qargs_val: HashMap<Qargs, PyObject> = HashMap::new();
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
                // // Obtain qarg hash for mapping
                // let qarg_hash = match qarg.hash() {
                //     Ok(vec) => vec,
                //     Err(e) => panic!(
                //         "Failed to hash q_args from the provided properties: {:?}.",
                //         e
                //     ),
                // };
                // Obtain values of qargs
                let qarg = match qarg.extract::<Vec<u32>>().ok() {
                    Some(vec) => Qargs { qargs: vec },
                    None => Qargs { qargs: vec![] },
                };
                // Store qargs hash value.
                // self.qarg_hash_table.insert(qarg_hash, qarg.clone());
                if !qarg.qargs.is_empty() && qarg.qargs.len() != instruction_num_qubits {
                    panic!("The number of qubits for {:?} does not match the number of qubits in the properties dictionary: {:?}", &instruction_name, qarg.qargs)
                }
                if !qarg.qargs.is_empty() {
                    self.num_qubits = self
                        .num_qubits
                        .max(qarg.qargs.iter().cloned().fold(0, u32::max) as usize + 1)
                }
                qargs_val.insert(qarg.clone(), values.to_object(py));
                if self.qarg_gate_map.contains_key(&qarg) {
                    self.qarg_gate_map
                        .get_mut(&qarg)
                        .unwrap()
                        .insert(instruction_name.clone());
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
        println!(
            "Before - {:?}: {:?}",
            instruction, self.gate_map[&instruction]
        );
        if !self.gate_map.contains_key(&instruction) {
            panic!(
                "Provided instruction : '{:?}' not in this Target.",
                &instruction
            );
        };
        let qargs = match qargs.extract::<Vec<u32>>().ok() {
            Some(vec) => Qargs { qargs: vec },
            None => Qargs { qargs: vec![] },
        };
        if !self.gate_map[&instruction].contains_key(&qargs) {
            panic!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs, &instruction
            );
        }
        self.gate_map.get_mut(&instruction).map(|q_vals| {
            *q_vals.get_mut(&qargs).unwrap() = properties;
            Some(())
        });
        self.instructions_durations = Option::None;
        self.instruction_schedule_map = Option::None;
        println!(
            "After - {:?}: {:?}",
            instruction, self.gate_map[&instruction]
        );
    }

    #[getter]
    fn instructions(&self) -> PyResult<Vec<(PyObject, Vec<u32>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Vec<u32>)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            for qarg in self.gate_map[op].keys() {
                let instruction_pair = (self.gate_name_map[op].clone(), qarg.qargs.clone());
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }

    #[pyo3(text_signature = "(/, inst_map, inst_name_map=None, error_dict=None")]
    fn update_from_instruction_schedule_map(
        &self,
        py: Python<'_>,
        inst_map: PyObject,
        inst_name_map: Option<Bound<PyDict>>,
        error_dict: Option<Bound<PyDict>>,
    ) {
        let get_calibration = match inst_map.getattr(py, "_get_calibration_entry") {
            Ok(calibration) => calibration,
            Err(e) => panic!(
                "Could not extract calibration from the provided Instruction map: {:?}",
                e
            ),
        };

        // Expand name mapping with custom gate namer provided by user.
        // TEMPORARY: Get arround improting this module and function, will be fixed after python wrapping.
        let qiskit_inst_name_map = import_from_module_call(
            py,
            "qiskit.circuit.library.standard_gates",
            "get_standard_gate_name_mapping",
            None,
        );

        // Update with inst_name_map if possible.
        if inst_name_map.is_some() {
            let _ = qiskit_inst_name_map.call_method1("update", (inst_name_map.unwrap(),));
        }

        while let Ok(inst_name) = inst_map.getattr(py, "instruction") {
            let inst_name = inst_name.extract::<String>(py).ok().unwrap();
            let mut out_props: HashMap<Qargs, PyObject> = HashMap::new();
            while let Ok(qargs) =
                inst_map.call_method1(py, "qubits_with_instruction", (&inst_name,))
            {
                let qargs = match qargs.extract::<Vec<u32>>(py).ok() {
                    Some(qargs) => Qargs { qargs },
                    None => Qargs {
                        qargs: vec![qargs.extract::<u32>(py).ok().unwrap()],
                    },
                };
                let mut props = self.gate_map[&inst_name].get(&qargs);
                let entry = match get_calibration.call1(py, (&inst_name, qargs.qargs.clone())) {
                    Ok(ent) => ent,
                    Err(e) => panic!(
                        "Could not obtain calibration with '{:?}' : {:?}.",
                        inst_name, e
                    ),
                };
                if entry.getattr(py, "user_provided").is_ok_and(|res| res.extract::<bool>(py).unwrap_or(false)) && 
                // TEMPORAL: Compare using __eq__
                !props.unwrap().getattr(py, "_calibration").unwrap().call_method1(py, "__eq__", (&entry,)).unwrap().extract::<bool>(py).unwrap_or(false)
                {
                    let duration: Option<f32>;
                    if self.dt.is_some() {
                        let entry_duration = entry
                            .call_method0(py, "get_schedule")
                            .unwrap()
                            .getattr(py, "duration");
                        duration = if entry_duration.is_ok() {
                            Some(
                                entry_duration.unwrap().extract::<f32>(py).unwrap()
                                    * self.dt.unwrap(),
                            )
                        } else {
                            None
                        }
                    } else {
                        duration = None;
                    }
                    // TODO: Create InstructionProperty for this
                    // props = Some(InstructionProperties::new(duration, None, Some(entry)).into_py(py));
                } else {
                    if props.is_none() {
                        // Edge case. Calibration is backend defined, but this is not
                        // registered in the backend target. Ignore this entry.
                        continue;
                    }
                }

                // TEMPORARY until a better way is found
                // WIP: Change python attributes from rust.

                // let gate_error = match &error_dict {
                //     Some(error_dic) => match error_dic.get_item(&inst_name).unwrap_or(None) {
                //         Some(gate) => match gate.downcast_into::<PyDict>().ok() {
                //             Some(gate_dict) => match gate_dict.get_item(qargs.qargs).ok() {
                //                 Some(error) => error,
                //                 None => None,
                //             },
                //             None => None,
                //         },
                //         None => None,
                //     },
                //     None => None,
                // };
                // // if gate_error.is_some() {
                // //     props.unwrap().error = gate_error.unwrap().into_py(py);
                // // }

                if let Some(x) = out_props.get_mut(&qargs) {
                    *x = props.unwrap().into_py(py)
                };
            }
        }
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
