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
    exceptions::PyTypeError,
    prelude::*,
    pyclass,
    types::{PyDict, PyList, PyTuple},
};

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

#[pyclass(module = "qiskit._accelerate.target.InstructionProperties")]
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f32>,
    #[pyo3(get, set)]
    pub error: Option<f32>,
    pub calibration: Option<PyObject>,
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
        }
    }

    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> Option<PyObject> {
        match &self.calibration {
            Some(calibration) => calibration.call_method0(py, "get_schedule").ok(),
            None => None,
        }
    }

    #[setter]
    pub fn set_calibration(&mut self, py: Python<'_>, calibration: Bound<PyAny>) -> PyResult<()> {
        self.calibration = Some(calibration.to_object(py));
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        if let Some(calibration) = self.get_calibration(py) {
            format!(
                "InstructionProperties(duration={:?}, error={:?}, calibration={:?})",
                self.duration,
                self.error,
                calibration.call_method0(py, "__repr__")
            )
        } else {
            format!(
                "InstructionProperties(duration={:?}, error={:?}, calibration=None)",
                self.duration, self.error
            )
        }
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
    #[pyo3(get)]
    gate_map: HashMap<String, HashMap<HashableVec<u32>, PyObject>>,
    #[pyo3(get)]
    gate_name_map: HashMap<String, PyObject>,
    global_operations: HashMap<usize, HashSet<String>>,
    qarg_gate_map: HashMap<HashableVec<u32>, HashSet<String>>,
    #[pyo3(get, set)]
    instruction_durations: Option<PyObject>,
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
            instruction_durations: Option::None,
            instruction_schedule_map: Option::None,
        }
    }

    #[pyo3(text_signature = "(/, instruction, properties=None, name=None")]
    fn add_instruction(
        &mut self,
        py: Python<'_>,
        instruction: PyObject,
        is_class: bool,
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
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
    }

    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, PyObject)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, PyObject)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            for qarg in self.gate_map[op].keys() {
                let instruction_pair = (self.gate_name_map[op].clone(), qarg.clone().into_py(py));
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }

    #[pyo3(text_signature = "/")]
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
            for (qarg, properties) in qargs.iter() {
                // Directly getting calibration entry to invoke .get_schedule().
                // This keeps PulseQobjDef unparsed.
                let cal_entry = properties.getattr(py, "_calibration").ok();
                if let Some(cal_entry) = cal_entry {
                    let _ = out_inst_schedule_map
                        .call_method1("_add", (instruction, qarg.clone(), cal_entry));
                }
            }
        }
        self.instruction_schedule_map = Some(out_inst_schedule_map.clone().unbind());
        out_inst_schedule_map.to_object(py)
    }

    #[getter]
    fn qargs(&self) -> PyResult<Option<HashSet<HashableVec<u32>>>> {
        let qargs: HashSet<HashableVec<u32>> = self.qarg_gate_map.clone().into_keys().collect();
        if qargs.len() == 1 && qargs.iter().next().is_none() {
            return Ok(None);
        }
        Ok(Some(qargs))
    }

    #[pyo3(text_signature = "(/, operation)")]
    fn qargs_for_operation_name(
        &self,
        operation: String,
    ) -> PyResult<Option<Vec<HashableVec<u32>>>> {
        /*
        Get the qargs for a given operation name

        Args:
           operation (str): The operation name to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to.
         */
        if self.gate_map[&operation].is_empty() {
            return Ok(None);
        }
        let qargs: Vec<HashableVec<u32>> = self.gate_map[&operation].clone().into_keys().collect();
        Ok(Some(qargs))
    }

    #[pyo3(text_signature = "(/, operation)")]
    fn operations_for_qargs(
        &self,
        py: Python<'_>,
        isclass: &Bound<PyAny>,
        qargs: Option<HashableVec<u32>>,
    ) -> PyResult<PyObject> {
        let res = PyList::empty_bound(py);
        if let Some(qargs) = qargs.clone() {
            if qargs
                .vec
                .iter()
                .any(|x| !(0..(self.num_qubits as u32)).contains(x))
            {
                // TODO: Throw Python Exception
                return Err(PyTypeError::new_err(format!(
                    "{:?} not in target.",
                    qargs.vec
                )));
            }

            for x in self.qarg_gate_map[&qargs].clone() {
                res.append(self.gate_name_map[&x].clone())?;
            }
            if let Some(qarg) = self.global_operations.get(&qargs.vec.len()) {
                for arg in qarg {
                    res.append(arg)?;
                }
            }
        }
        for op in self.gate_name_map.values() {
            if isclass.call1((op,))?.extract::<bool>()? {
                res.append(op)?;
            }
        }
        if res.is_empty() {
            if let Some(qargs) = qargs {
                return Err(PyTypeError::new_err(format!(
                    "{:?} not in target",
                    qargs.vec
                )));
            } else {
                return Err(PyTypeError::new_err(format!("{:?} not in target", qargs)));
            }
        }
        Ok(res.to_object(py))
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
