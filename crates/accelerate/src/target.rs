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
    exceptions::{PyAttributeError, PyIndexError, PyKeyError},
    prelude::*,
    pyclass,
    types::{IntoPyDict, PyDict, PyList, PySequence, PyTuple, PyType},
};

use self::exceptions::TranspilerError;

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

    import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
    import_exception_bound! {qiskit.providers.exceptions, BackendPropertyError}
}

#[pyclass(module = "qiskit._accelerate.target.InstructionProperties")]
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f32>,
    #[pyo3(get, set)]
    pub error: Option<f32>,
    #[pyo3(get)]
    _calibration: Option<PyObject>,
}

#[pymethods]
impl InstructionProperties {
    #[new]
    #[pyo3(text_signature = "(/, duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,)")]
    pub fn new(duration: Option<f32>, error: Option<f32>, calibration: Option<PyObject>) -> Self {
        InstructionProperties {
            error,
            duration,
            _calibration: calibration,
        }
    }

    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> Option<PyObject> {
        match &self._calibration {
            Some(calibration) => calibration.call_method0(py, "get_schedule").ok(),
            None => None,
        }
    }

    #[setter]
    pub fn set_calibration(&mut self, py: Python<'_>, calibration: Bound<PyAny>) -> PyResult<()> {
        self._calibration = Some(calibration.to_object(py));
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if let Some(calibration) = self.get_calibration(py) {
            Ok(format!(
                "InstructionProperties(duration={:?}, error={:?}, calibration={:?})",
                self.duration,
                self.error,
                calibration
                    .call_method0(py, "__repr__")?
                    .extract::<String>(py)?
            ))
        } else {
            Ok(format!(
                "InstructionProperties(duration={:?}, error={:?}, calibration=None)",
                self.duration, self.error
            ))
        }
    }
}

type GateMapType = HashMap<String, Option<HashMap<Option<HashableVec<u32>>, Option<PyObject>>>>;

#[pyclass(mapping, module = "qiskit._accelerate.target.Target")]
#[derive(Clone, Debug)]
pub struct Target {
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get)]
    pub num_qubits: Option<usize>,
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
    pub concurrent_measurements: Vec<Vec<usize>>,
    // Maybe convert PyObjects into rust representations of Instruction and Data
    #[pyo3(get)]
    gate_map: GateMapType,
    #[pyo3(get)]
    gate_name_map: HashMap<String, PyObject>,
    global_operations: HashMap<usize, HashSet<String>>,
    #[pyo3(get)]
    qarg_gate_map: HashMap<Option<HashableVec<u32>>, Option<HashSet<String>>>,
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
        dt: Option<f32>,
        granularity: Option<i32>,
        min_length: Option<usize>,
        pulse_alignment: Option<i32>,
        acquire_alignment: Option<i32>,
        qubit_properties: Option<Vec<PyObject>>,
        concurrent_measurements: Option<Vec<Vec<usize>>>,
    ) -> Self {
        Target {
            description: description.unwrap_or("".to_string()),
            num_qubits,
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
        instruction: PyObject,
        is_class: bool,
        properties: Option<HashMap<Option<HashableVec<u32>>, Option<PyObject>>>,
        name: Option<String>,
    ) -> PyResult<()> {
        // Unwrap instruction name
        let instruction_name: String;
        let mut properties = properties;
        if !is_class {
            if let Some(name) = name {
                instruction_name = name;
            } else {
                instruction_name = instruction.getattr(py, "name")?.extract::<String>(py)?;
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
            properties = Some(HashMap::from_iter([(None, None)].into_iter()));
        }
        if self.gate_map.contains_key(&instruction_name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {:?} is already in the target",
                instruction_name
            )));
        }
        self.gate_name_map
            .insert(instruction_name.clone(), instruction.clone());
        let mut qargs_val: HashMap<Option<HashableVec<u32>>, Option<PyObject>> = HashMap::new();
        if is_class {
            qargs_val = HashMap::from_iter([(None, None)].into_iter());
        } else if let Some(properties) = properties {
            let inst_num_qubits = instruction
                .getattr(py, "num_qubits")?
                .extract::<usize>(py)?;
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
                    .or_insert(Some(HashSet::from_iter(
                        [instruction_name.clone()].into_iter(),
                    )));
            }
        }
        self.gate_map.insert(instruction_name, Some(qargs_val));
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
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
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        let qargs = HashableVec { vec: qargs };
        if let Some(gate_map_instruction) = self.gate_map[&instruction].as_ref() {
            if !gate_map_instruction.contains_key(&Some(qargs.clone())) {
                return Err(PyKeyError::new_err(format!(
                    "Provided qarg {:?} not in this Target for {:?}.",
                    &qargs.vec, &instruction
                )));
            }
        }
        if let Some(Some(q_vals)) = self.gate_map.get_mut(&instruction) {
            if let Some(q_vals) = q_vals.get_mut(&Some(qargs)) {
                *q_vals = Some(properties);
            }
        }
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
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
            if let Some(qargs) = qargs {
                for (qarg, properties) in qargs.iter() {
                    // Directly getting calibration entry to invoke .get_schedule().
                    // This keeps PulseQobjDef unparsed.
                    if let Some(properties) = properties {
                        let cal_entry = properties.getattr(py, "_calibration").ok();
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
        if let Some(gate_map_oper) = self.gate_map[&operation].as_ref() {
            if gate_map_oper.is_empty() {
                return Ok(None);
            }

            let qargs: Vec<Option<HashableVec<u32>>> =
                gate_map_oper.to_owned().into_keys().collect();
            return Ok(Some(qargs));
        }
        Ok(Some(vec![]))
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn operations_for_qargs(
        &self,
        py: Python<'_>,
        isclass: &Bound<PyAny>,
        qargs: Option<HashableVec<u32>>,
    ) -> PyResult<Py<PyList>> {
        let res = PyList::empty_bound(py);
        for op in self.gate_name_map.values() {
            if isclass.call1((op,))?.extract::<bool>()? {
                res.append(op)?;
            }
        }
        if let Some(qargs) = qargs {
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

            if let Some(gate_map_qarg) = self.qarg_gate_map[&Some(qargs.clone())].clone() {
                for x in gate_map_qarg {
                    res.append(self.gate_name_map[&x].clone())?;
                }
            }

            if let Some(qarg) = self.global_operations.get(&qargs.vec.len()) {
                for arg in qarg {
                    res.append(arg)?;
                }
            }
            if res.is_empty() {
                return Err(PyKeyError::new_err(format!(
                    "{:?} not in target",
                    qargs.vec
                )));
            }
        }
        Ok(res.into())
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn operation_names_for_qargs(
        &self,
        isclass: &Bound<PyAny>,
        qargs: Option<HashableVec<u32>>,
    ) -> PyResult<HashSet<String>> {
        // When num_qubits == 0 we return globally defined operators
        let mut res = HashSet::new();
        let mut qargs = qargs;
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(qargs) = qargs {
            if qargs
                .vec
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default() as u32).contains(x))
            {
                return Err(PyKeyError::new_err(format!("{:?}", qargs)));
            }
            if let Some(qarg_gate_map_arg) = self.qarg_gate_map[&Some(qargs.clone())].clone() {
                res.extend(qarg_gate_map_arg);
            }
            if let Some(ext) = self.global_operations.get(&qargs.vec.len()) {
                res = ext.union(&res).cloned().collect();
            }
            for (name, op) in self.gate_name_map.iter() {
                if isclass.call1((op,))?.extract::<bool>()? {
                    res.insert(name.into());
                }
            }
            if res.is_empty() {
                return Err(PyKeyError::new_err(format!(
                    "{:?} not in target",
                    qargs.vec
                )));
            }
        }
        Ok(res)
    }

    #[pyo3(text_signature = "(/, qargs)")]
    fn instruction_supported(
        &self,
        py: Python<'_>,
        isclass: &Bound<PyAny>,
        isinstance: &Bound<PyAny>,
        parameter_class: &Bound<PyAny>,
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
                    if isclass.call1((obj,))?.extract::<bool>()? {
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

                    if isinstance
                        .call1((obj, operation_class))?
                        .extract::<bool>()?
                    {
                        if let Some(parameters) = parameters {
                            if parameters.len()
                                != obj
                                    .getattr(py, "params")?
                                    .downcast_bound::<PySequence>(py)?
                                    .len()?
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
                            || self.gate_map[op_name]
                                .as_ref()
                                .unwrap_or(&HashMap::from_iter([(None, None)].into_iter()))
                                .contains_key(&None)
                        {
                            let qubit_comparison = self.gate_name_map[op_name]
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
                return Ok(false);
            }
            if let Some(operation_name) = operation_name {
                if self.gate_map.contains_key(&operation_name) {
                    let mut obj = &self.gate_name_map[&operation_name];
                    if isclass.call1((obj,))?.extract::<bool>()? {
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
                            if isinstance
                                .call1((obj_params.get_item(index)?, parameter_class))?
                                .extract::<bool>()?
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
                        if gate_map_oper.contains_key(&Some(qargs_.clone())) {
                            return Ok(true);
                        }
                    }

                    // Double check this
                    if self.gate_map[&operation_name].is_none()
                        || self.gate_map[&operation_name]
                            .as_ref()
                            .unwrap_or(&HashMap::from_iter([(None, None)].into_iter()))
                            .contains_key(&None)
                    {
                        obj = &self.gate_name_map[&operation_name];
                        if isclass.call1((obj,))?.extract::<bool>()? {
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
    fn has_calibration(
        &self,
        py: Python<'_>,
        operation_name: String,
        qargs: HashableVec<u32>,
    ) -> PyResult<bool> {
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
                return Ok(!oper_qarg.getattr(py, "_calibration")?.is_none(py));
            } else {
                return Ok(false);
            }
        }
        Ok(false)
    }

    #[pyo3(text_signature = "( /, operation_name: str, qargs: tuple[int, ...],)")]
    fn get_calibration(
        &self,
        py: Python<'_>,
        operation_name: String,
        qargs: HashableVec<u32>,
        args: Bound<PyTuple>,
        kwargs: Option<Bound<PyDict>>,
    ) -> PyResult<PyObject> {
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
        if !self.has_calibration(py, operation_name.clone(), qargs.clone())? {
            return Err(PyKeyError::new_err(format!(
                "Calibration of instruction {:?} for qubit {:?} is not defined.",
                operation_name, qargs.vec
            )));
        }
        Ok(
            self.gate_map[&operation_name].as_ref().unwrap()[&Some(qargs)]
                .as_ref()
                .unwrap()
                .getattr(py, "_calibration")?
                .call_method_bound(py, "get_schedule", args, kwargs.as_ref())?
                .to_object(py),
        )
    }

    #[pyo3(text_signature = "(/, index: int)")]
    fn instruction_properties(&self, index: usize) -> PyResult<PyObject> {
        let mut instruction_properties: Vec<PyObject> = vec![];
        for operation in self.gate_map.keys() {
            if let Some(gate_map_oper) = self.gate_map[operation].to_owned() {
                for (_, inst_props) in gate_map_oper.iter() {
                    if let Some(inst_props) = inst_props {
                        instruction_properties.push(inst_props.to_owned())
                    }
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
    fn get_non_global_operation_names(
        &mut self,
        py: Python<'_>,
        strict_direction: bool,
    ) -> PyResult<PyObject> {
        let mut search_set: HashSet<HashableVec<u32>> = HashSet::new();
        if strict_direction {
            if let Some(global_strict) = &self.non_global_strict_basis {
                return Ok(global_strict.to_object(py));
            }
            // Build search set
            for qarg_key in self.qarg_gate_map.keys().flatten().cloned() {
                search_set.insert(qarg_key);
            }
        } else {
            if let Some(global_basis) = &self.non_global_basis {
                return Ok(global_basis.to_object(py));
            }
            for qarg_key in self.qarg_gate_map.keys().flatten().cloned() {
                if qarg_key.vec.len() != 1 {
                    search_set.insert(qarg_key);
                }
            }
        }
        let mut incomplete_basis_gates: Vec<String> = vec![];
        let mut size_dict: HashMap<usize, usize> = HashMap::new();
        *size_dict
            .entry(1)
            .or_insert(self.num_qubits.unwrap_or_default()) = self.num_qubits.unwrap_or_default();
        for qarg in search_set {
            if qarg.vec.len() == 1 {
                continue;
            }
            *size_dict.entry(qarg.vec.len()).or_insert(0) += 1;
        }
        for (inst, qargs) in self.gate_map.iter() {
            if let Some(qargs) = qargs {
                let mut qarg_len = qargs.len();
                let qarg_sample = qargs.keys().next();
                if qarg_sample.is_none() {
                    continue;
                }
                let qarg_sample = qarg_sample.unwrap();
                if !strict_direction {
                    let mut qarg_set = HashSet::new();
                    for qarg in qargs.keys() {
                        if let Some(qarg) = qarg.to_owned() {
                            qarg_set.insert(qarg);
                        }
                    }
                    qarg_len = qarg_set.len();
                }
                if let Some(qarg_sample) = qarg_sample {
                    if qarg_len != size_dict[&qarg_sample.vec.len()] {
                        incomplete_basis_gates.push(inst.to_owned());
                    }
                }
            }
        }
        if strict_direction {
            self.non_global_strict_basis = Some(incomplete_basis_gates);
            Ok(self.non_global_strict_basis.to_object(py))
        } else {
            self.non_global_basis = Some(incomplete_basis_gates);
            Ok(self.non_global_basis.to_object(py))
        }
    }

    // Class properties
    #[getter]
    fn qargs(&self) -> PyResult<Option<HashSet<Option<HashableVec<u32>>>>> {
        let qargs: HashSet<Option<HashableVec<u32>>> =
            self.qarg_gate_map.clone().into_keys().collect();
        if qargs.len() == 1 && qargs.iter().next().is_none() {
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
    fn operation_names(&self) -> Vec<String> {
        // Get the operation names in the target.
        return Vec::from_iter(self.gate_map.keys().cloned());
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

    #[classmethod]
    fn from_configuration(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        qubits_props_list_from_props: Bound<PyAny>,
        get_standard_gate_name_mapping: Bound<PyAny>,
        isclass: Bound<PyAny>,
        basis_gates: Vec<String>,
        num_qubits: Option<usize>,
        coupling_map: Option<PyObject>,
        inst_map: Option<Bound<PyAny>>,
        backend_properties: Option<&Bound<PyAny>>,
        instruction_durations: Option<PyObject>,
        concurrent_measurements: Option<Vec<Vec<usize>>>,
        dt: Option<f32>,
        timing_constraints: Option<PyObject>,
        custom_name_mapping: Option<HashMap<String, Bound<PyAny>>>,
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
                    } else if isclass.call1((&gate_obj,))?.extract::<bool>()? {
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
                let mut gate_properties: HashMap<Option<HashableVec<u32>>, Option<PyObject>> =
                    HashMap::new();
                for qubit in 0..num_qubits.unwrap_or_default() {
                    let mut error: Option<f32> = None;
                    let mut duration: Option<f32> = None;
                    let mut calibration: Option<PyObject> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, qubit))?
                                .extract::<f32>()
                            {
                                Ok(duration) => Some(duration),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties
                            .call_method1("gate_error", (&gate, qubit))?
                            .extract::<f32>()
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
                                            .extract::<f32>()?
                                            * dt.unwrap_or_default(),
                                    );
                                }
                                Some(calibration.to_object(py))
                            }
                            Err(_) => None,
                        }
                    }
                    if let Some(instruction_durations) = &instruction_durations {
                        let kwargs = [("unit", "s")].into_py_dict_bound(py);
                        duration = match instruction_durations
                            .call_method_bound(py, "get", (&gate, qubit), Some(&kwargs))?
                            .extract::<f32>(py)
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
                            Some(
                                InstructionProperties::new(duration, error, calibration)
                                    .into_py(py),
                            ),
                        );
                    }
                }
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(
                        py,
                        inst.unbind(),
                        isclass
                            .call1((name_mapping.get_item(&gate)?,))?
                            .extract::<bool>()?,
                        Some(gate_properties),
                        Some(gate),
                    )?;
                }
            }
            let edges = coupling_map
                .call_method0(py, "get_edges")?
                .extract::<Vec<[u32; 2]>>(py)?;
            for gate in two_qubit_gates {
                let mut gate_properties: HashMap<Option<HashableVec<u32>>, Option<PyObject>> =
                    HashMap::new();
                for edge in edges.as_slice().iter().copied() {
                    let mut error: Option<f32> = None;
                    let mut duration: Option<f32> = None;
                    let mut calibration: Option<PyObject> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, edge))?
                                .extract::<f32>()
                            {
                                Ok(duration) => Some(duration),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties
                            .call_method1("gate_error", (&gate, edge))?
                            .extract::<f32>()
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
                                            .extract::<f32>()?
                                            * dt.unwrap_or_default(),
                                    );
                                }
                                Some(calibration.to_object(py))
                            }
                            Err(_) => None,
                        }
                    }
                    if let Some(instruction_durations) = &instruction_durations {
                        let kwargs = [("unit", "s")].into_py_dict_bound(py);
                        duration = match instruction_durations
                            .call_method_bound(py, "get", (&gate, edge), Some(&kwargs))?
                            .extract::<f32>(py)
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
                            Some(
                                InstructionProperties::new(duration, error, calibration)
                                    .into_py(py),
                            ),
                        );
                    }
                }
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(
                        py,
                        inst.unbind(),
                        isclass
                            .call1((name_mapping.get_item(&gate)?,))?
                            .extract::<bool>()?,
                        Some(gate_properties),
                        Some(gate),
                    )?;
                }
            }
            for gate in global_ideal_variable_width_gates {
                if let Some(inst) = name_mapping.get_item(&gate)? {
                    target.add_instruction(
                        py,
                        inst.unbind(),
                        isclass
                            .call1((name_mapping.get_item(&gate)?,))?
                            .extract::<bool>()?,
                        None,
                        Some(gate),
                    )?;
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
                    target.add_instruction(
                        py,
                        inst.unbind(),
                        isclass
                            .call1((name_mapping.get_item(&gate)?,))?
                            .extract::<bool>()?,
                        None,
                        Some(gate),
                    )?;
                }
            }
        }
        Ok(target)
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
