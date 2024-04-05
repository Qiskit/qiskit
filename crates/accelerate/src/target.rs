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
use pyo3::{
    prelude::*,
    pyclass,
    types::{PyDict, PyTuple},
};

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
        // Unwrap properties
        let properties: Bound<PyDict> = properties.unwrap_or(PyDict::new_bound(py));
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
        // Cont. unwrap instruction name
        if instruction_name.is_empty() {
            instruction_name = match instruction.getattr(py, "name") {
                Ok(i_name) => match i_name.extract::<String>(py) {
                    Ok(i_name) => {
                        // TODO: Figure out how to identify whether a class is received or not
                        // if !properties.is_empty() {
                        //     panic!("An instruction added globally by class can't have properties set.");
                        // };
                        i_name
                    },
                    Err(e) => panic!("The provided instruction does not have a valid 'name' attribute: {:?}", e)
                },
                Err(e) => panic!("A name must be specified when defining a supported global operation by class: {:?}", e)
            };
        }
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
        // TODO: Introduce a similar case to is_class

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
        let mut qargs_val: HashMap<isize, PyObject> = HashMap::new();
        for (qarg, values) in properties {
            // Obtain source qargs
            let qarg = match qarg.downcast::<PyTuple>() {
                Ok(tuple) => tuple,
                Err(e) => panic!(
                    "Failed to downcast q_args from the provided properties: {:?}.",
                    e
                ),
            };
            // Obtain qarg hash for mapping
            let qarg_hash = match qarg.hash() {
                Ok(vec) => vec,
                Err(e) => panic!(
                    "Failed to hash q_args from the provided properties: {:?}.",
                    e
                ),
            };
            // Obtain values of qargs
            let qarg = match qarg.extract::<Vec<usize>>() {
                Ok(vec) => vec,
                Err(e) => panic!(
                    "Failed to extract q_args from the provided properties: {:?}.",
                    e
                ),
            };
            // Store qargs hash value.
            self.qarg_hash_table.insert(qarg_hash, qarg.clone());
            println!("{:?}: {:?}", qarg, qarg_hash);
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
        self.gate_map.insert(instruction_name, qargs_val);
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
    m.add_class::<Target>()?;
    Ok(())
}
