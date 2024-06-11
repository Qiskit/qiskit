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

mod errors;
mod instruction_properties;

use std::ops::Index;

use ahash::RandomState;
use hashbrown::HashSet;
use indexmap::{
    map::{Keys, Values},
    IndexMap, IndexSet,
};
use itertools::Itertools;
use pyo3::{
    exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyValueError},
    prelude::*,
    pyclass,
    types::{PyDict, PyType},
};

use smallvec::{smallvec, SmallVec};

use crate::nlayout::PhysicalQubit;

use errors::TargetKeyError;
use instruction_properties::InstructionProperties;

use self::exceptions::TranspilerError;

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.exceptions, QiskitError}
    import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
}

// Custom types
type Qargs = SmallVec<[PhysicalQubit; 2]>;
type GateMap = IndexMap<String, PropsMap, RandomState>;
type PropsMap = IndexMap<Option<Qargs>, Option<InstructionProperties>, RandomState>;
type GateMapState = Vec<(String, Vec<(Option<Qargs>, Option<InstructionProperties>)>)>;

/// Temporary interpretation of Gate
#[derive(Debug, Clone)]
pub struct GateRep {
    pub object: PyObject,
    pub num_qubits: Option<usize>,
    pub label: Option<String>,
    pub params: Option<Vec<Param>>,
}

impl FromPyObject<'_> for GateRep {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let num_qubits = match ob.getattr("num_qubits") {
            Ok(num_qubits) => num_qubits.extract::<usize>().ok(),
            Err(_) => None,
        };
        let label = match ob.getattr("label") {
            Ok(label) => label.extract::<String>().ok(),
            Err(_) => None,
        };
        let params = match ob.getattr("params") {
            Ok(params) => params.extract::<Vec<Param>>().ok(),
            Err(_) => None,
        };
        Ok(Self {
            object: ob.into(),
            num_qubits,
            label,
            params,
        })
    }
}

impl IntoPy<PyObject> for GateRep {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.object
    }
}

// Temporary interpretation of Param
#[derive(Debug, Clone, FromPyObject)]
pub enum Param {
    Float(f64),
    ParameterExpression(PyObject),
}

// Temporary interpretation of Python Parameter
impl Param {
    fn compare(one: &PyObject, other: &PyObject) -> bool {
        Python::with_gil(|py| -> PyResult<bool> {
            let other_bound = other.bind(py);
            other_bound.eq(one)
        })
        .unwrap()
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Param::Float(s), Param::Float(other)) => s == other,
            (Param::Float(_), Param::ParameterExpression(_)) => false,
            (Param::ParameterExpression(_), Param::Float(_)) => false,
            (Param::ParameterExpression(s), Param::ParameterExpression(other)) => {
                Self::compare(s, other)
            }
        }
    }
}

/**
A rust representation of a ``Target`` object.

The intent of the ``Target`` object is to inform Qiskit's compiler about
the constraints of a particular backend so the compiler can compile an
input circuit to something that works and is optimized for a device. It
currently contains a description of instructions on a backend and their
properties as well as some timing information. However, this exact
interface may evolve over time as the needs of the compiler change. These
changes will be done in a backwards compatible and controlled manner when
they are made (either through versioning, subclassing, or mixins) to add
on to the set of information exposed by a target.
 */
#[pyclass(
    mapping,
    subclass,
    name = "BaseTarget",
    module = "qiskit._accelerate.target"
)]
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
    // TODO: Port to Rust.
    pub qubit_properties: Option<Vec<PyObject>>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Vec<Vec<usize>>,
    gate_map: GateMap,
    #[pyo3(get)]
    _gate_name_map: IndexMap<String, GateRep, RandomState>,
    global_operations: IndexMap<usize, HashSet<String>, RandomState>,
    variable_class_operations: IndexSet<String, RandomState>,
    qarg_gate_map: IndexMap<Option<Qargs>, Option<HashSet<String>>, RandomState>,
    non_global_strict_basis: Option<Vec<String>>,
    non_global_basis: Option<Vec<String>>,
}

#[pymethods]
impl Target {
    /// Create a new ``Target`` object
    ///
    ///Args:
    ///    description (str): An optional string to describe the Target.
    ///    num_qubits (int): An optional int to specify the number of qubits
    ///        the backend target has. If not set it will be implicitly set
    ///        based on the qargs when :meth:`~qiskit.Target.add_instruction`
    ///        is called. Note this must be set if the backend target is for a
    ///        noiseless simulator that doesn't have constraints on the
    ///        instructions so the transpiler knows how many qubits are
    ///        available.
    ///    dt (float): The system time resolution of input signals in seconds
    ///    granularity (int): An integer value representing minimum pulse gate
    ///        resolution in units of ``dt``. A user-defined pulse gate should
    ///        have duration of a multiple of this granularity value.
    ///    min_length (int): An integer value representing minimum pulse gate
    ///        length in units of ``dt``. A user-defined pulse gate should be
    ///        longer than this length.
    ///    pulse_alignment (int): An integer value representing a time
    ///        resolution of gate instruction starting time. Gate instruction
    ///        should start at time which is a multiple of the alignment
    ///        value.
    ///    acquire_alignment (int): An integer value representing a time
    ///        resolution of measure instruction starting time. Measure
    ///        instruction should start at time which is a multiple of the
    ///        alignment value.
    ///    qubit_properties (list): A list of :class:`~.QubitProperties`
    ///        objects defining the characteristics of each qubit on the
    ///        target device. If specified the length of this list must match
    ///        the number of qubits in the target, where the index in the list
    ///        matches the qubit number the properties are defined for. If some
    ///        qubits don't have properties available you can set that entry to
    ///        ``None``
    ///    concurrent_measurements(list): A list of sets of qubits that must be
    ///        measured together. This must be provided
    ///        as a nested list like ``[[0, 1], [2, 3, 4]]``.
    ///Raises:
    ///    ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
    ///        defined and the value of ``num_qubits`` differs from the length of
    ///        ``qubit_properties``.
    #[new]
    #[pyo3(signature = (
        description = None,
        num_qubits = None,
        dt = None,
        granularity = None,
        min_length = None,
        pulse_alignment = None,
        acquire_alignment = None,
        qubit_properties = None,
        concurrent_measurements = None,
    ))]
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
    ) -> PyResult<Self> {
        let mut num_qubits = num_qubits;
        if let Some(qubit_properties) = qubit_properties.as_ref() {
            if let Some(num_qubits) = num_qubits {
                if num_qubits != qubit_properties.len() {
                    return Err(PyValueError::new_err(
                        "The value of num_qubits specified does not match the \
                            length of the input qubit_properties list",
                    ));
                }
            } else {
                num_qubits = Some(qubit_properties.len())
            }
        }
        Ok(Target {
            description,
            num_qubits,
            dt,
            granularity: granularity.unwrap_or(1),
            min_length: min_length.unwrap_or(1),
            pulse_alignment: pulse_alignment.unwrap_or(1),
            acquire_alignment: acquire_alignment.unwrap_or(0),
            qubit_properties,
            concurrent_measurements: concurrent_measurements.unwrap_or(Vec::new()),
            gate_map: GateMap::default(),
            _gate_name_map: IndexMap::default(),
            variable_class_operations: IndexSet::default(),
            global_operations: IndexMap::default(),
            qarg_gate_map: IndexMap::default(),
            non_global_basis: None,
            non_global_strict_basis: None,
        })
    }

    /// Add a new instruction to the :class:`~qiskit.transpiler.Target`
    ///
    /// As ``Target`` objects are strictly additive this is the primary method
    /// for modifying a ``Target``. Typically, you will use this to fully populate
    /// a ``Target`` before using it in :class:`~qiskit.providers.BackendV2`. For
    /// example::
    ///
    ///     from qiskit.circuit.library import CXGate
    ///     from qiskit.transpiler import Target, InstructionProperties
    ///
    ///     target = Target()
    ///     cx_properties = {
    ///         (0, 1): None,
    ///         (1, 0): None,
    ///         (0, 2): None,
    ///         (2, 0): None,
    ///         (0, 3): None,
    ///         (2, 3): None,
    ///         (3, 0): None,
    ///         (3, 2): None
    ///     }
    ///     target.add_instruction(CXGate(), cx_properties)
    ///
    /// Will add a :class:`~qiskit.circuit.library.CXGate` to the target with no
    /// properties (duration, error, etc) with the coupling edge list:
    /// ``(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (2, 3), (3, 0), (3, 2)``. If
    /// there are properties available for the instruction you can replace the
    /// ``None`` value in the properties dictionary with an
    /// :class:`~qiskit.transpiler.InstructionProperties` object. This pattern
    /// is repeated for each :class:`~qiskit.circuit.Instruction` the target
    /// supports.
    ///
    /// Args:
    ///     instruction (Union[qiskit.circuit.Instruction, Type[qiskit.circuit.Instruction]]):
    ///         The operation object to add to the map. If it's parameterized any value
    ///         of the parameter can be set. Optionally for variable width
    ///         instructions (such as control flow operations such as :class:`~.ForLoop` or
    ///         :class:`~MCXGate`) you can specify the class. If the class is specified than the
    ///         ``name`` argument must be specified. When a class is used the gate is treated as global
    ///         and not having any properties set.
    ///     properties (dict): A dictionary of qarg entries to an
    ///         :class:`~qiskit.transpiler.InstructionProperties` object for that
    ///         instruction implementation on the backend. Properties are optional
    ///         for any instruction implementation, if there are no
    ///         :class:`~qiskit.transpiler.InstructionProperties` available for the
    ///         backend the value can be None. If there are no constraints on the
    ///         instruction (as in a noiseless/ideal simulation) this can be set to
    ///         ``{None, None}`` which will indicate it runs on all qubits (or all
    ///         available permutations of qubits for multi-qubit gates). The first
    ///         ``None`` indicates it applies to all qubits and the second ``None``
    ///         indicates there are no
    ///         :class:`~qiskit.transpiler.InstructionProperties` for the
    ///         instruction. By default, if properties is not set it is equivalent to
    ///         passing ``{None: None}``.
    ///     name (str): An optional name to use for identifying the instruction. If not
    ///         specified the :attr:`~qiskit.circuit.Instruction.name` attribute
    ///         of ``gate`` will be used. All gates in the ``Target`` need unique
    ///         names. Backends can differentiate between different
    ///         parameterization of a single gate by providing a unique name for
    ///         each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
    ///         documentation for the :class:`~qiskit.transpiler.Target` class).
    /// Raises:
    ///     AttributeError: If gate is already in map
    ///     TranspilerError: If an operation class is passed in for ``instruction`` and no name
    ///         is specified or ``properties`` is set.
    #[pyo3(signature = (instruction, name, is_class, properties=None))]
    fn add_instruction(
        &mut self,
        _py: Python<'_>,
        instruction: GateRep,
        name: String,
        is_class: bool,
        properties: Option<PropsMap>,
    ) -> PyResult<()> {
        if self.gate_map.contains_key(&name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {:?} is already in the target",
                name
            )));
        }
        let mut qargs_val: PropsMap = PropsMap::default();
        if is_class {
            qargs_val = IndexMap::from_iter([(None, None)].into_iter());
            self.variable_class_operations.insert(name.clone());
        } else if let Some(properties) = properties {
            let inst_num_qubits = instruction.num_qubits.unwrap_or_default();
            if properties.contains_key(&None) {
                self.global_operations
                    .entry(inst_num_qubits)
                    .and_modify(|e| {
                        e.insert(name.clone());
                    })
                    .or_insert(HashSet::from_iter([name.clone()]));
            }
            for qarg in properties.keys() {
                let mut qarg_obj = None;
                if let Some(qarg) = qarg {
                    if qarg.len() != inst_num_qubits {
                        return Err(TranspilerError::new_err(format!(
                            "The number of qubits for {name} does not match\
                             the number of qubits in the properties dictionary: {:?}",
                            qarg
                        )));
                    }
                    self.num_qubits =
                        Some(self.num_qubits.unwrap_or_default().max(
                            qarg.iter().fold(
                                0,
                                |acc, x| {
                                    if acc > x.index() {
                                        acc
                                    } else {
                                        x.index()
                                    }
                                },
                            ) + 1,
                        ));
                    qarg_obj = Some(qarg.clone())
                }
                qargs_val.insert(qarg_obj.to_owned(), properties[qarg].clone());
                self.qarg_gate_map
                    .entry(qarg_obj)
                    .and_modify(|e| {
                        if let Some(e) = e {
                            e.insert(name.clone());
                        }
                    })
                    .or_insert(Some(HashSet::from([name.clone()])));
            }
        }
        // TODO: Modify logic once gates are in rust.
        self._gate_name_map.insert(name.clone(), instruction);
        self.gate_map.insert(name, qargs_val);
        self.non_global_basis = None;
        self.non_global_strict_basis = None;
        Ok(())
    }

    /// Update the property object for an instruction qarg pair already in the Target
    ///
    /// Args:
    ///     instruction (str): The instruction name to update
    ///     qargs (tuple): The qargs to update the properties of
    ///     properties (InstructionProperties): The properties to set for this instruction
    /// Raises:
    ///     KeyError: If ``instruction`` or ``qarg`` are not in the target
    #[pyo3(text_signature = "(instruction, qargs, properties, /,)")]
    fn update_instruction_properties(
        &mut self,
        instruction: String,
        qargs: Option<Qargs>,
        properties: Option<InstructionProperties>,
    ) -> PyResult<()> {
        if !self.contains_key(&instruction) {
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        let mut prop_map = self[&instruction].clone();
        if !(prop_map.contains_key(&qargs)) {
            return Err(PyKeyError::new_err(format!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs.unwrap_or_default(),
                &instruction
            )));
        }
        prop_map.entry(qargs).and_modify(|e| *e = properties);
        self.gate_map
            .entry(instruction)
            .and_modify(|e| *e = prop_map);
        Ok(())
    }

    /// Get the qargs for a given operation name
    ///
    /// Args:
    ///     operation (str): The operation name to get qargs for
    /// Returns:
    ///     set: The set of qargs the gate instance applies to.
    #[pyo3(text_signature = "(operation, /,)")]
    fn qargs_for_operation_name(&self, operation: String) -> PyResult<Option<Vec<Qargs>>> {
        match self.qargs_for_op_name(&operation) {
            Ok(option_set) => match option_set {
                Some(set) => Ok(Some(set.into_iter().cloned().collect())),
                None => Ok(None),
            },
            Err(e) => Err(PyKeyError::new_err(e.message)),
        }
    }

    /// Get the operation class object for a given name
    ///
    /// Args:
    ///     instruction (str): The instruction name to get the
    ///         :class:`~qiskit.circuit.Instruction` instance for
    /// Returns:
    ///     qiskit.circuit.Instruction: The Instruction instance corresponding to the
    ///     name. This also can also be the class for globally defined variable with
    ///     operations.
    #[pyo3(text_signature = "(instruction, /)")]
    pub fn operation_from_name(&self, py: Python<'_>, instruction: String) -> PyResult<PyObject> {
        if let Some(gate_obj) = self._gate_name_map.get(&instruction) {
            Ok(gate_obj.object.clone_ref(py))
        } else {
            Err(PyKeyError::new_err(format!(
                "Instruction {:?} not in target",
                instruction
            )))
        }
    }

    /// Get the operation class object for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A qargs tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return any globally defined operations in the target.
    /// Returns:
    ///     list: The list of :class:`~qiskit.circuit.Instruction` instances
    ///     that apply to the specified qarg. This may also be a class if
    ///     a variable width operation is globally defined.
    ///
    /// Raises:
    ///     KeyError: If qargs is not in target
    #[pyo3(text_signature = "(/, qargs=None)")]
    fn operations_for_qargs(
        &self,
        py: Python<'_>,
        qargs: Option<Qargs>,
    ) -> PyResult<Vec<PyObject>> {
        // Move to rust native once Gates are in rust
        Ok(self
            .operation_names_for_qargs(qargs)?
            .into_iter()
            .map(|x| self._gate_name_map[x].object.clone_ref(py))
            .collect())
    }

    /// Get the operation names for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return the names for any globally defined operations in the target.
    /// Returns:
    ///     set: The set of operation names that apply to the specified ``qargs``.
    ///
    /// Raises:
    ///     KeyError: If ``qargs`` is not in target
    #[pyo3(text_signature = "(/, qargs=None)")]
    pub fn operation_names_for_qargs(
        &self,
        qargs: Option<Qargs>,
    ) -> PyResult<HashSet<&String, RandomState>> {
        match self.op_names_for_qargs(&qargs) {
            Ok(set) => Ok(set),
            Err(e) => Err(PyKeyError::new_err(e.message)),
        }
    }

    /// Return whether the instruction (operation + qubits) is supported by the target
    ///
    /// Args:
    ///     operation_name (str): The name of the operation for the instruction. Either
    ///         this or ``operation_class`` must be specified, if both are specified
    ///         ``operation_class`` will take priority and this argument will be ignored.
    ///     qargs (tuple): The tuple of qubit indices for the instruction. If this is
    ///         not specified then this method will return ``True`` if the specified
    ///         operation is supported on any qubits. The typical application will
    ///         always have this set (otherwise it's the same as just checking if the
    ///         target contains the operation). Normally you would not set this argument
    ///         if you wanted to check more generally that the target supports an operation
    ///         with the ``parameters`` on any qubits.
    ///     operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
    ///         the target supports a particular operation by class rather
    ///         than by name. This lookup is more expensive as it needs to
    ///         iterate over all operations in the target instead of just a
    ///         single lookup. If this is specified it will supersede the
    ///         ``operation_name`` argument. The typical use case for this
    ///         operation is to check whether a specific variant of an operation
    ///         is supported on the backend. For example, if you wanted to
    ///         check whether a :class:`~.RXGate` was supported on a specific
    ///         qubit with a fixed angle. That fixed angle variant will
    ///         typically have a name different from the object's
    ///         :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
    ///         This can be used to check if any instances of the class are
    ///         available in such a case.
    ///     parameters (list): A list of parameters to check if the target
    ///         supports them on the specified qubits. If the instruction
    ///         supports the parameter values specified in the list on the
    ///         operation and qargs specified this will return ``True`` but
    ///         if the parameters are not supported on the specified
    ///         instruction it will return ``False``. If this argument is not
    ///         specified this method will return ``True`` if the instruction
    ///         is supported independent of the instruction parameters. If
    ///         specified with any :class:`~.Parameter` objects in the list,
    ///         that entry will be treated as supporting any value, however parameter names
    ///         will not be checked (for example if an operation in the target
    ///         is listed as parameterized with ``"theta"`` and ``"phi"`` is
    ///         passed into this function that will return ``True``). For
    ///         example, if called with::
    ///
    ///             parameters = [Parameter("theta")]
    ///             target.instruction_supported("rx", (0,), parameters=parameters)
    ///
    ///         will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
    ///         that will accept any parameter. If you need to check for a fixed numeric
    ///         value parameter this argument is typically paired with the ``operation_class``
    ///         argument. For example::
    ///
    ///             target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])
    ///
    ///         will return ``True`` if an RXGate(pi/4) exists on qubit 0.
    ///
    /// Returns:
    ///     bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.
    #[pyo3(
        text_signature = "(/, operation_name=None, qargs=None, operation_class=None, parameters=None)"
    )]
    pub fn instruction_supported(
        &self,
        py: Python<'_>,
        operation_name: Option<String>,
        qargs: Option<Qargs>,
        operation_class: Option<&Bound<PyAny>>,
        parameters: Option<Vec<Param>>,
    ) -> PyResult<bool> {
        // Do this in case we need to modify qargs
        let mut qargs = qargs;

        // Check obj param function
        let check_obj_params = |parameters: &Vec<Param>, obj: &GateRep| -> PyResult<bool> {
            for (index, param) in parameters.iter().enumerate() {
                let param_at_index = &obj.params.as_ref().map(|x| &x[index]).unwrap();
                match (param, param_at_index) {
                    (Param::Float(p1), Param::Float(p2)) => {
                        if *p1 != *p2 {
                            return Ok(false);
                        }
                    }
                    (&Param::Float(_), Param::ParameterExpression(_)) => continue,
                    (&Param::ParameterExpression(_), Param::Float(_)) => return Ok(false),
                    (Param::ParameterExpression(_), Param::ParameterExpression(_)) => continue,
                }
            }
            Ok(true)
        };

        if self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(operation_class) = operation_class {
            for (op_name, obj) in self._gate_name_map.iter() {
                if self.variable_class_operations.contains(op_name) {
                    if !operation_class.eq(&obj.object)? {
                        continue;
                    }
                    // If no qargs operation class is supported
                    if let Some(_qargs) = &qargs {
                        let qarg_set: HashSet<PhysicalQubit> = _qargs.iter().cloned().collect();
                        // If qargs set then validate no duplicates and all indices are valid on device
                        if _qargs
                            .iter()
                            .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                            && qarg_set.len() == _qargs.len()
                        {
                            return Ok(true);
                        } else {
                            return Ok(false);
                        }
                    } else {
                        return Ok(true);
                    }
                }

                if obj
                    .object
                    .bind_borrowed(py)
                    .is_instance(operation_class.downcast::<PyType>()?)?
                {
                    if let Some(parameters) = &parameters {
                        if parameters.len()
                            != obj.params.as_ref().map(|x| x.len()).unwrap_or_default()
                        {
                            continue;
                        }
                        if !check_obj_params(parameters, obj)? {
                            continue;
                        }
                    }
                    if let Some(_qargs) = &qargs {
                        if self.gate_map.contains_key(op_name) {
                            let gate_map_name = &self.gate_map[op_name];
                            if gate_map_name.contains_key(&qargs) {
                                return Ok(true);
                            }
                            if gate_map_name.contains_key(&None) {
                                let qubit_comparison =
                                    self._gate_name_map[op_name].num_qubits.unwrap_or_default();
                                return Ok(qubit_comparison == _qargs.len()
                                    && _qargs
                                        .iter()
                                        .all(|x| x.index() < self.num_qubits.unwrap_or_default()));
                            }
                        } else {
                            let qubit_comparison = obj.num_qubits.unwrap_or_default();
                            return Ok(qubit_comparison == _qargs.len()
                                && _qargs
                                    .iter()
                                    .all(|x| x.index() < self.num_qubits.unwrap_or_default()));
                        }
                    } else {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }

        if let Some(operation_names) = &operation_name {
            if self.gate_map.contains_key(operation_names) {
                if let Some(parameters) = parameters {
                    let obj = self._gate_name_map[operation_names].to_owned();
                    if self.variable_class_operations.contains(operation_names) {
                        if let Some(_qargs) = qargs {
                            let qarg_set: HashSet<PhysicalQubit> = _qargs.iter().cloned().collect();
                            if _qargs
                                .iter()
                                .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                && qarg_set.len() == _qargs.len()
                            {
                                return Ok(true);
                            } else {
                                return Ok(false);
                            }
                        } else {
                            return Ok(true);
                        }
                    }

                    let obj_params = obj.params.unwrap_or_default();
                    if parameters.len() != obj_params.len() {
                        return Ok(false);
                    }
                    for (index, params) in parameters.iter().enumerate() {
                        let mut matching_params = false;
                        let obj_at_index = &obj_params[index];
                        if matches!(obj_at_index, Param::ParameterExpression(_))
                            || params == &obj_params[index]
                        {
                            matching_params = true;
                        }
                        if !matching_params {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                if let Some(_qargs) = qargs.as_ref() {
                    let qarg_set: HashSet<PhysicalQubit> = _qargs.iter().cloned().collect();
                    if let Some(gate_prop_name) = self.gate_map.get(operation_names) {
                        if gate_prop_name.contains_key(&qargs) {
                            return Ok(true);
                        }
                        if gate_prop_name.contains_key(&None) {
                            let obj = &self._gate_name_map[operation_names];
                            if self.variable_class_operations.contains(operation_names) {
                                if qargs.is_none()
                                    || _qargs.iter().all(|qarg| {
                                        qarg.index() <= self.num_qubits.unwrap_or_default()
                                    }) && qarg_set.len() == _qargs.len()
                                {
                                    return Ok(true);
                                } else {
                                    return Ok(false);
                                }
                            } else {
                                let qubit_comparison = obj.num_qubits.unwrap_or_default();
                                return Ok(qubit_comparison == _qargs.len()
                                    && _qargs.iter().all(|qarg| {
                                        qarg.index() < self.num_qubits.unwrap_or_default()
                                    }));
                            }
                        }
                    } else {
                        // Duplicate case is if it contains none
                        if self.variable_class_operations.contains(operation_names) {
                            if qargs.is_none()
                                || _qargs
                                    .iter()
                                    .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                    && qarg_set.len() == _qargs.len()
                            {
                                return Ok(true);
                            } else {
                                return Ok(false);
                            }
                        } else {
                            let qubit_comparison = self._gate_name_map[operation_names]
                                .num_qubits
                                .unwrap_or_default();
                            return Ok(qubit_comparison == _qargs.len()
                                && _qargs.iter().all(|qarg| {
                                    qarg.index() < self.num_qubits.unwrap_or_default()
                                }));
                        }
                    }
                } else {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Get the instruction properties for a specific instruction tuple
    ///
    /// This method is to be used in conjunction with the
    /// :attr:`~qiskit.transpiler.Target.instructions` attribute of a
    /// :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
    /// get the instruction properties for an element of
    /// :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
    /// However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
    /// directly it is likely more efficient to access the target directly via the name
    /// and qubits to get the instruction properties. For example, if
    /// :attr:`~qiskit.transpiler.Target.instructions` returned::
    ///
    ///     [(XGate(), (0,)), (XGate(), (1,))]
    ///
    /// you could get the properties of the ``XGate`` on qubit 1 with::
    ///
    ///     props = target.instruction_properties(1)
    ///
    /// but just accessing it directly via the name would be more efficient::
    ///
    ///     props = target['x'][(1,)]
    ///
    /// (assuming the ``XGate``'s canonical name in the target is ``'x'``)
    /// This is especially true for larger targets as this will scale worse with the number
    /// of instruction tuples in a target.
    ///
    /// Args:
    ///     index (int): The index of the instruction tuple from the
    ///         :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
    ///         if you want the properties from the third element in
    ///         :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
    /// Returns:
    ///     InstructionProperties: The instruction properties for the specified instruction tuple
    #[pyo3(text_signature = "(/, index: int)")]
    pub fn instruction_properties(
        &self,
        _py: Python<'_>,
        index: usize,
    ) -> PyResult<Option<InstructionProperties>> {
        let mut index_counter = 0;
        for (_operation, props_map) in self.gate_map.iter() {
            let gate_map_oper = props_map.values();
            for inst_props in gate_map_oper {
                if index_counter == index {
                    return Ok(inst_props.to_owned());
                }
                index_counter += 1;
            }
        }
        Err(PyIndexError::new_err(format!(
            "Index: {:?} is out of range.",
            index
        )))
    }

    /// Return the non-global operation names for the target
    ///
    /// The non-global operations are those in the target which don't apply
    /// on all qubits (for single qubit operations) or all multi-qubit qargs
    /// (for multi-qubit operations).
    ///
    /// Args:
    ///     strict_direction (bool): If set to ``True`` the multi-qubit
    ///         operations considered as non-global respect the strict
    ///         direction (or order of qubits in the qargs is significant). For
    ///         example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
    ///         defined over ``(1, 0)`` by default neither would be considered
    ///         non-global, but if ``strict_direction`` is set ``True`` both
    ///         ``cx`` and ``ecr`` would be returned.
    ///
    /// Returns:
    ///     List[str]: A list of operation names for operations that aren't global in this target
    #[pyo3(signature = (/, strict_direction=false,), text_signature = "(/, strict_direction=False)")]
    pub fn get_non_global_operation_names(
        &mut self,
        py: Python<'_>,
        strict_direction: bool,
    ) -> PyObject {
        self.get_non_global_op_names(strict_direction).to_object(py)
    }

    // Class properties

    /// The set of qargs in the target.
    #[getter]
    fn qargs(&self) -> Option<Vec<Option<Qargs>>> {
        self.get_qargs()
            .map(|qargs| qargs.into_iter().cloned().collect())
    }

    /// Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
    /// for the target
    ///
    /// For globally defined variable width operations the tuple will be of the form
    /// ``(class, None)`` where class is the actual operation class that
    /// is globally defined.
    #[getter]
    pub fn instructions(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, Option<Qargs>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Option<Qargs>)> = vec![];
        // Add all operations and dehash qargs.
        for (op, props_map) in self.gate_map.iter() {
            for qarg in props_map.keys() {
                let instruction_pair = (self._gate_name_map[op].object.clone_ref(py), qarg.clone());
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }
    /// Get the operation names in the target.
    #[getter]
    pub fn operation_names(&self) -> Vec<String> {
        self.gate_map.keys().cloned().collect()
    }

    /// Get the operation objects in the target.
    #[getter]
    pub fn operations(&self) -> Vec<&PyObject> {
        return Vec::from_iter(self._gate_name_map.values().map(|x| &x.object));
    }

    /// Returns a sorted list of physical qubits.
    #[getter]
    pub fn physical_qubits(&self) -> Vec<usize> {
        Vec::from_iter(0..self.num_qubits.unwrap_or_default())
    }

    // Magic methods:

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result_list = PyDict::new_bound(py);
        result_list.set_item("description", self.description.clone())?;
        result_list.set_item("num_qubits", self.num_qubits)?;
        result_list.set_item("dt", self.dt)?;
        result_list.set_item("granularity", self.granularity)?;
        result_list.set_item("min_length", self.min_length)?;
        result_list.set_item("pulse_alignment", self.pulse_alignment)?;
        result_list.set_item("acquire_alignment", self.acquire_alignment)?;
        result_list.set_item("qubit_properties", self.qubit_properties.clone())?;
        result_list.set_item(
            "concurrent_measurements",
            self.concurrent_measurements.clone(),
        )?;
        result_list.set_item(
            "gate_map",
            self.gate_map
                .clone()
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        value
                            .into_iter()
                            .collect::<Vec<(Option<Qargs>, Option<InstructionProperties>)>>(),
                    )
                })
                .collect::<GateMapState>()
                .into_py(py),
        )?;
        result_list.set_item("gate_name_map", self._gate_name_map.clone().into_py(py))?;
        result_list.set_item("global_operations", self.global_operations.clone())?;
        result_list.set_item(
            "qarg_gate_map",
            self.qarg_gate_map.clone().into_iter().collect_vec(),
        )?;
        result_list.set_item("non_global_basis", self.non_global_basis.clone())?;
        result_list.set_item(
            "non_global_strict_basis",
            self.non_global_strict_basis.clone(),
        )?;
        Ok(result_list.unbind())
    }

    fn __setstate__(&mut self, state: Bound<PyDict>) -> PyResult<()> {
        self.description = state
            .get_item("description")?
            .unwrap()
            .extract::<Option<String>>()?;
        self.num_qubits = state
            .get_item("num_qubits")?
            .unwrap()
            .extract::<Option<usize>>()?;
        self.dt = state.get_item("dt")?.unwrap().extract::<Option<f64>>()?;
        self.granularity = state.get_item("granularity")?.unwrap().extract::<i32>()?;
        self.min_length = state.get_item("min_length")?.unwrap().extract::<usize>()?;
        self.pulse_alignment = state
            .get_item("pulse_alignment")?
            .unwrap()
            .extract::<i32>()?;
        self.acquire_alignment = state
            .get_item("acquire_alignment")?
            .unwrap()
            .extract::<i32>()?;
        self.qubit_properties = state
            .get_item("qubit_properties")?
            .unwrap()
            .extract::<Option<Vec<PyObject>>>()?;
        self.concurrent_measurements = state
            .get_item("concurrent_measurements")?
            .unwrap()
            .extract::<Vec<Vec<usize>>>()?;
        self.gate_map = IndexMap::from_iter(
            state
                .get_item("gate_map")?
                .unwrap()
                .extract::<GateMapState>()?
                .into_iter()
                .map(|(name, prop_map)| (name, IndexMap::from_iter(prop_map.into_iter()))),
        );
        self._gate_name_map = state
            .get_item("gate_name_map")?
            .unwrap()
            .extract::<IndexMap<String, GateRep, RandomState>>()?;
        self.global_operations = state
            .get_item("global_operations")?
            .unwrap()
            .extract::<IndexMap<usize, HashSet<String>, RandomState>>()?;
        self.qarg_gate_map = IndexMap::from_iter(
            state
                .get_item("qarg_gate_map")?
                .unwrap()
                .extract::<Vec<(Option<Qargs>, Option<HashSet<String>>)>>()?,
        );
        self.non_global_basis = state
            .get_item("non_global_basis")?
            .unwrap()
            .extract::<Option<Vec<String>>>()?;
        self.non_global_strict_basis = state
            .get_item("non_global_strict_basis")?
            .unwrap()
            .extract::<Option<Vec<String>>>()?;
        Ok(())
    }
}

// Rust native methods
impl Target {
    /// Generate non global operations if missing
    fn generate_non_global_op_names(&mut self, strict_direction: bool) -> &Vec<String> {
        let mut search_set: HashSet<Qargs, RandomState> = HashSet::default();
        if strict_direction {
            // Build search set
            for qarg_key in self.qarg_gate_map.keys().flatten().cloned() {
                search_set.insert(qarg_key);
            }
        } else {
            for qarg_key in self.qarg_gate_map.keys().flatten() {
                if qarg_key.len() != 1 {
                    let mut vec = qarg_key.clone();
                    vec.sort_unstable();
                    search_set.insert(vec);
                }
            }
        }
        let mut incomplete_basis_gates: Vec<String> = vec![];
        let mut size_dict: IndexMap<usize, usize, RandomState> = IndexMap::default();
        *size_dict
            .entry(1)
            .or_insert(self.num_qubits.unwrap_or_default()) = self.num_qubits.unwrap_or_default();
        for qarg in &search_set {
            if qarg.len() == 1 {
                continue;
            }
            *size_dict.entry(qarg.len()).or_insert(0) += 1;
        }
        for (inst, qargs_props) in self.gate_map.iter() {
            let mut qarg_len = qargs_props.len();
            let qargs_keys: IndexSet<&Option<Qargs>> = qargs_props.keys().collect();
            let qarg_sample = qargs_keys.iter().next().cloned();
            if let Some(qarg_sample) = qarg_sample {
                if !strict_direction {
                    let mut qarg_set: HashSet<SmallVec<[PhysicalQubit; 2]>, RandomState> =
                        HashSet::default();
                    for qarg in qargs_keys {
                        let mut qarg_set_vec: Qargs = smallvec![];
                        if let Some(qarg) = qarg {
                            let mut to_vec = qarg.to_owned();
                            to_vec.sort_unstable();
                            qarg_set_vec = to_vec;
                        }
                        qarg_set.insert(qarg_set_vec);
                    }
                    qarg_len = qarg_set.len();
                }
                if let Some(qarg_sample) = qarg_sample {
                    if qarg_len != *size_dict.entry(qarg_sample.len()).or_insert(0) {
                        incomplete_basis_gates.push(inst.to_owned());
                    }
                }
            }
        }
        if strict_direction {
            self.non_global_strict_basis = Some(incomplete_basis_gates);
            self.non_global_strict_basis.as_ref().unwrap()
        } else {
            self.non_global_basis = Some(incomplete_basis_gates.to_owned());
            self.non_global_basis.as_ref().unwrap()
        }
    }

    /// Get all non_global operation names.
    pub fn get_non_global_op_names(&mut self, strict_direction: bool) -> Option<&Vec<String>> {
        if strict_direction {
            if self.non_global_strict_basis.is_some() {
                return self.non_global_strict_basis.as_ref();
            }
        } else if self.non_global_basis.is_some() {
            return self.non_global_basis.as_ref();
        }
        return Some(self.generate_non_global_op_names(strict_direction));
    }

    /// Gets all the operation names that use these qargs. Rust native equivalent of ``BaseTarget.operation_names_for_qargs()``
    pub fn op_names_for_qargs(
        &self,
        qargs: &Option<Qargs>,
    ) -> Result<HashSet<&String, RandomState>, TargetKeyError> {
        // When num_qubits == 0 we return globally defined operators
        let mut res: HashSet<&String, RandomState> = HashSet::default();
        let mut qargs = qargs;
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = &None;
        }
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
            {
                return Err(TargetKeyError::new_err(format!(
                    "{:?} not in Target",
                    qargs
                )));
            }
        }
        if let Some(Some(qarg_gate_map_arg)) = self.qarg_gate_map.get(qargs).as_ref() {
            res.extend(qarg_gate_map_arg);
        }
        for name in self._gate_name_map.keys() {
            if self.variable_class_operations.contains(name) {
                res.insert(name);
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(global_gates) = self.global_operations.get(&qargs.len()) {
                res.extend(global_gates)
            }
        }
        if res.is_empty() {
            return Err(TargetKeyError::new_err(format!(
                "{:?} not in target",
                qargs
            )));
        }
        Ok(res)
    }

    /// Gets all the qargs used by the specified operation name. Rust native equivalent of ``BaseTarget.qargs_for_operation_name()``
    pub fn qargs_for_op_name(
        &self,
        operation: &String,
    ) -> Result<Option<Vec<&Qargs>>, TargetKeyError> {
        if let Some(gate_map_oper) = self.gate_map.get(operation) {
            if gate_map_oper.contains_key(&None) {
                return Ok(None);
            }
            let qargs: Vec<&Qargs> = gate_map_oper.keys().flatten().collect();
            Ok(Some(qargs))
        } else {
            Err(TargetKeyError::new_err(format!(
                "Operation: {operation} not in Target."
            )))
        }
    }

    /// Rust-native method to get all the qargs of a specific Target object
    pub fn get_qargs(&self) -> Option<IndexSet<&Option<Qargs>>> {
        let qargs: IndexSet<&Option<Qargs>> = self.qarg_gate_map.keys().collect();
        // TODO: Modify logic to account for the case of {None}
        let next_entry = qargs.iter().next();
        if qargs.len() == 1
            && (qargs.first().unwrap().is_none()
                || next_entry.is_none()
                || next_entry.unwrap().is_none())
        {
            return None;
        }
        Some(qargs)
    }

    // IndexMap methods

    /// Retreive all the gate names in the Target
    pub fn keys(&self) -> Keys<String, PropsMap> {
        self.gate_map.keys()
    }

    /// Retrieves an iterator over the property maps stored within the Target
    pub fn values(&self) -> Values<String, PropsMap> {
        self.gate_map.values()
    }

    /// Checks if a key exists in the Target
    pub fn contains_key(&self, key: &String) -> bool {
        self.gate_map.contains_key(key)
    }
}

// To access the Target's gate map by gate name.
impl Index<&str> for Target {
    type Output = PropsMap;
    fn index(&self, index: &str) -> &Self::Output {
        self.gate_map.index(index)
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
