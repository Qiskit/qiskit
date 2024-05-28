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

mod instruction_properties;

use hashbrown::HashSet;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use pyo3::{
    exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyValueError},
    prelude::*,
    pyclass,
    sync::GILOnceCell,
    types::{PyList, PyType},
};
use smallvec::{smallvec, SmallVec};

use crate::nlayout::PhysicalQubit;

use instruction_properties::BaseInstructionProperties;

use self::exceptions::TranspilerError;

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.exceptions, QiskitError}
    import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
}

/// Import isclass function from python.
static ISCLASS: GILOnceCell<PyObject> = GILOnceCell::new();

/// Import parameter class from python.
static PARAMETER_CLASS: GILOnceCell<Py<PyType>> = GILOnceCell::new();

/// Helper function to import inspect.isclass from python.
fn isclass(object: &Bound<PyAny>) -> PyResult<bool> {
    ISCLASS
        .get_or_init(object.py(), || -> PyObject {
            Python::with_gil(|py| -> PyResult<PyObject> {
                let inspect_module: Bound<PyModule> = py.import_bound("inspect")?;
                Ok(inspect_module.getattr("isclass")?.into())
            })
            .unwrap()
        })
        .call1(object.py(), (object,))?
        .extract::<bool>(object.py())
}

fn get_parameter(py: Python<'_>) -> PyResult<&Py<PyType>> {
    Ok(PARAMETER_CLASS.get_or_init(py, || -> Py<PyType> {
        Python::with_gil(|py| -> PyResult<Py<PyType>> {
            let parameter_class = py
                .import_bound("qiskit.circuit.parameter")?
                .getattr("Parameter")?;
            Ok(parameter_class.downcast::<PyType>()?.clone().unbind())
        })
        .unwrap()
    }))
}

pub fn tupleize(py: Python<'_>, qargs: Qargs) -> PyObject {
    match qargs.len() {
        1 => qargs
            .into_iter()
            .collect_tuple::<(PhysicalQubit,)>()
            .to_object(py),
        2 => qargs
            .into_iter()
            .collect_tuple::<(PhysicalQubit, PhysicalQubit)>()
            .to_object(py),
        3 => qargs
            .into_iter()
            .collect_tuple::<(PhysicalQubit, PhysicalQubit, PhysicalQubit)>()
            .to_object(py),
        4 => qargs
            .into_iter()
            .collect_tuple::<(PhysicalQubit, PhysicalQubit, PhysicalQubit, PhysicalQubit)>()
            .to_object(py),
        _ => py.None(),
    }
}

// Custom types
type Qargs = SmallVec<[PhysicalQubit; 4]>;
type GateMap = IndexMap<String, PropsMap>;
type PropsMap = IndexMap<Option<Qargs>, Option<BaseInstructionProperties>>;
type GateMapState = Vec<(
    String,
    Vec<(Option<Qargs>, Option<BaseInstructionProperties>)>,
)>;

// Temporary interpretation of Param
#[derive(Debug, Clone, FromPyObject)]
enum Param {
    Float(f64),
    ParameterExpression(PyObject),
}

/**
The intent of the ``Target`` object is to inform Qiskit's compiler about
the constraints of a particular backend so the compiler can compile an
input circuit to something that works and is optimized for a device. It
currently contains a description of instructions on a backend and their
properties as well as some timing information. However, this exact
interface may evolve over time as the needs of the compiler change. These
changes will be done in a backwards compatible and controlled manner when
they are made (either through versioning, subclassing, or mixins) to add
on to the set of information exposed by a target.

As a basic example, let's assume backend has two qubits, supports
:class:`~qiskit.circuit.library.UGate` on both qubits and
:class:`~qiskit.circuit.library.CXGate` in both directions. To model this
you would create the target like::

    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import UGate, CXGate
    from qiskit.circuit import Parameter

    gmap = Target()
    theta = Parameter('theta')
    phi = Parameter('phi')
    lam = Parameter('lambda')
    u_props = {
        (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
        (1,): InstructionProperties(duration=4.52e-8, error=0.00032115),
    }
    gmap.add_instruction(UGate(theta, phi, lam), u_props)
    cx_props = {
        (0,1): InstructionProperties(duration=5.23e-7, error=0.00098115),
        (1,0): InstructionProperties(duration=4.52e-7, error=0.00132115),
    }
    gmap.add_instruction(CXGate(), cx_props)

Each instruction in the ``Target`` is indexed by a unique string name that uniquely
identifies that instance of an :class:`~qiskit.circuit.Instruction` object in
the Target. There is a 1:1 mapping between a name and an
:class:`~qiskit.circuit.Instruction` instance in the target and each name must
be unique. By default, the name is the :attr:`~qiskit.circuit.Instruction.name`
attribute of the instruction, but can be set to anything. This lets a single
target have multiple instances of the same instruction class with different
parameters. For example, if a backend target has two instances of an
:class:`~qiskit.circuit.library.RXGate` one is parameterized over any theta
while the other is tuned up for a theta of pi/6 you can add these by doing something
like::

    import math

    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import RXGate
    from qiskit.circuit import Parameter

    target = Target()
    theta = Parameter('theta')
    rx_props = {
        (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
    }
    target.add_instruction(RXGate(theta), rx_props)
    rx_30_props = {
        (0,): InstructionProperties(duration=1.74e-6, error=.00012)
    }
    target.add_instruction(RXGate(math.pi / 6), rx_30_props, name='rx_30')

Then in the ``target`` object accessing by ``rx_30`` will get the fixed
angle :class:`~qiskit.circuit.library.RXGate` while ``rx`` will get the
parameterized :class:`~qiskit.circuit.library.RXGate`.

.. note::

    This class assumes that qubit indices start at 0 and are a contiguous
    set if you want a submapping the bits will need to be reindexed in
    a new``Target`` object.

.. note::

    This class only supports additions of gates, qargs, and qubits.
    If you need to remove one of these the best option is to iterate over
    an existing object and create a new subset (or use one of the methods
    to do this). The object internally caches different views and these
    would potentially be invalidated by removals.
 */
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
    pub qubit_properties: Option<Vec<PyObject>>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Vec<Vec<usize>>,
    gate_map: GateMap,
    #[pyo3(get)]
    _gate_name_map: IndexMap<String, PyObject>,
    global_operations: IndexMap<usize, HashSet<String>>,
    qarg_gate_map: IndexMap<Option<Qargs>, Option<HashSet<String>>>,
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
            gate_map: GateMap::new(),
            _gate_name_map: IndexMap::new(),
            global_operations: IndexMap::new(),
            qarg_gate_map: IndexMap::new(),
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
        instruction: &Bound<PyAny>,
        name: String,
        is_class: bool,
        properties: Option<PropsMap>,
    ) -> PyResult<()> {
        // Unwrap instruction name
        let properties = properties;

        if self.gate_map.contains_key(&name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {:?} is already in the target",
                name
            )));
        }
        self._gate_name_map
            .insert(name.clone(), instruction.clone().unbind());
        let mut qargs_val: PropsMap = PropsMap::new();
        if is_class {
            qargs_val = IndexMap::from_iter([(None, None)].into_iter());
        } else if let Some(properties) = properties {
            let inst_num_qubits = instruction.getattr("num_qubits")?.extract::<usize>()?;
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
                            "The number of qubits for {instruction} does not match\
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
        properties: Option<BaseInstructionProperties>,
    ) -> PyResult<()> {
        if !self.gate_map.contains_key(&instruction) {
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        let mut prop_map = self.gate_map[&instruction].clone();
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
        if let Some(gate_map_oper) = self.gate_map.get(&operation) {
            if gate_map_oper.contains_key(&None) {
                return Ok(None);
            }
            let qargs: Vec<Qargs> = gate_map_oper.keys().flatten().cloned().collect();
            Ok(Some(qargs))
        } else {
            Err(PyKeyError::new_err(format!(
                "Operation: {operation} not in Target."
            )))
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
    fn operation_from_name(&self, py: Python<'_>, instruction: String) -> PyResult<PyObject> {
        if let Some(gate_obj) = self._gate_name_map.get(&instruction) {
            Ok(gate_obj.clone_ref(py))
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
        let mut res: Vec<PyObject> = vec![];
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
            {
                return Err(PyKeyError::new_err(format!("{:?} not in target.", qargs)));
            }
        }
        if let Some(Some(gate_map_qarg)) = self.qarg_gate_map.get(&qargs) {
            for x in gate_map_qarg {
                res.push(self._gate_name_map[x].clone_ref(py));
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(inst_set) = self.global_operations.get(&qargs.len()) {
                for inst in inst_set {
                    res.push(self._gate_name_map[inst].clone_ref(py));
                }
            }
        }
        for (name, op) in self._gate_name_map.iter() {
            if self.gate_map[name].contains_key(&None) {
                res.push(op.clone_ref(py));
            }
        }
        if res.is_empty() {
            return Err(PyKeyError::new_err(format!("{:?} not in target", {
                match &qargs {
                    Some(qarg) => format!("{:?}", qarg),
                    None => "None".to_owned(),
                }
            })));
        }
        Ok(res)
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
    fn operation_names_for_qargs(
        &self,
        _py: Python<'_>,
        qargs: Option<Qargs>,
    ) -> PyResult<HashSet<&String>> {
        // When num_qubits == 0 we return globally defined operators
        let mut res = HashSet::new();
        let mut qargs = qargs;
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
            {
                return Err(PyKeyError::new_err(format!("{:?}", qargs)));
            }
        }
        if let Some(Some(qarg_gate_map_arg)) = self.qarg_gate_map.get(&qargs).as_ref() {
            res.extend(qarg_gate_map_arg);
        }
        for name in self._gate_name_map.keys() {
            if self.gate_map[name].contains_key(&None) {
                res.insert(name);
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(global_gates) = self.global_operations.get(&qargs.len()) {
                res.extend(global_gates)
            }
        }
        if res.is_empty() {
            return Err(PyKeyError::new_err(format!("{:?} not in target", qargs)));
        }
        Ok(res)
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
    fn instruction_supported(
        &self,
        py: Python<'_>,
        operation_name: Option<String>,
        qargs: Option<Qargs>,
        operation_class: Option<&Bound<PyAny>>,
        parameters: Option<&Bound<PyList>>,
    ) -> PyResult<bool> {
        // Do this in case we need to modify qargs
        let mut qargs = qargs;
        let parameter_class = get_parameter(py)?.bind(py);

        // Check obj param function
        let check_obj_params = |parameters: &Bound<PyList>, obj: &Bound<PyAny>| -> PyResult<bool> {
            for (index, param) in parameters.iter().enumerate() {
                let param_at_index = obj
                    .getattr("params")?
                    .downcast::<PyList>()?
                    .get_item(index)?;
                if param.is_instance(parameter_class)?
                    && !param_at_index.is_instance(parameter_class)?
                {
                    return Ok(false);
                }
                if !param.eq(&param_at_index)? && !param_at_index.is_instance(parameter_class)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };

        if self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(operation_class) = operation_class {
            for (op_name, obj) in self._gate_name_map.iter() {
                if isclass(obj.bind(py))? {
                    if !operation_class.eq(obj)? {
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
                        if !check_obj_params(parameters, obj.bind(py))? {
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
                                let qubit_comparison = self._gate_name_map[op_name]
                                    .getattr(py, "num_qubits")?
                                    .extract::<usize>(py)?;
                                return Ok(qubit_comparison == _qargs.len()
                                    && _qargs
                                        .iter()
                                        .all(|x| x.index() < self.num_qubits.unwrap_or_default()));
                            }
                        } else {
                            let qubit_comparison =
                                obj.getattr(py, "num_qubits")?.extract::<usize>(py)?;
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
                    if isclass(obj.bind(py))? {
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

                    let obj_params = obj.getattr(py, "params")?;
                    let obj_params = obj_params.downcast_bound::<PyList>(py)?;
                    if parameters.len() != obj_params.len() {
                        return Ok(false);
                    }
                    for (index, params) in parameters.iter().enumerate() {
                        let mut matching_params = false;
                        if obj_params.get_item(index)?.is_instance(parameter_class)?
                            || params.eq(obj_params.get_item(index)?)?
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
                            if isclass(obj.bind(py))? {
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
                                let qubit_comparison =
                                    obj.getattr(py, "num_qubits")?.extract::<usize>(py)?;
                                return Ok(qubit_comparison == _qargs.len()
                                    && _qargs.iter().all(|qarg| {
                                        qarg.index() < self.num_qubits.unwrap_or_default()
                                    }));
                            }
                        }
                    } else {
                        // Duplicate case is if it contains none
                        let obj = &self._gate_name_map[operation_names];
                        if isclass(obj.bind(py))? {
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
                                .getattr(py, "num_qubits")?
                                .extract::<usize>(py)?;
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
    fn instruction_properties(
        &self,
        _py: Python<'_>,
        index: usize,
    ) -> PyResult<Option<BaseInstructionProperties>> {
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
    fn get_non_global_operation_names(
        &mut self,
        _py: Python<'_>,
        strict_direction: bool,
    ) -> PyResult<Vec<String>> {
        let mut search_set: HashSet<Option<Qargs>> = HashSet::new();
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
            for qarg_key in self.qarg_gate_map.keys().flatten() {
                if qarg_key.len() != 1 {
                    let mut vec = qarg_key.clone();
                    vec.sort();
                    let qarg_key = Some(vec);
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
            if qarg.is_none() || qarg.as_ref().unwrap_or(&smallvec![]).len() == 1 {
                continue;
            }
            *size_dict
                .entry(qarg.to_owned().unwrap_or_default().len())
                .or_insert(0) += 1;
        }
        for (inst, qargs_props) in self.gate_map.iter() {
            let mut qarg_len = qargs_props.len();
            let qargs_keys: IndexSet<&Option<Qargs>> = qargs_props.keys().collect();
            let qarg_sample = qargs_keys.iter().next().cloned();
            if let Some(qarg_sample) = qarg_sample {
                if !strict_direction {
                    let mut qarg_set = HashSet::new();
                    for qarg in qargs_keys {
                        let mut qarg_set_vec: Qargs = smallvec![];
                        if let Some(qarg) = qarg {
                            let mut to_vec = qarg.to_owned();
                            to_vec.sort();
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
            self.non_global_strict_basis = Some(incomplete_basis_gates.to_owned());
            Ok(incomplete_basis_gates)
        } else {
            self.non_global_basis = Some(incomplete_basis_gates.to_owned());
            Ok(incomplete_basis_gates)
        }
    }

    // Class properties

    /// The set of qargs in the target.
    #[getter]
    fn qargs(&self) -> PyResult<Option<Vec<Qargs>>> {
        let qargs: Vec<Option<Qargs>> = self.qarg_gate_map.keys().cloned().collect();
        // Modify logic to account for the case of {None}
        let next_entry = qargs.iter().flatten().next();
        if qargs.len() == 1 && (qargs.first().unwrap().is_none() || next_entry.is_none()) {
            return Ok(None);
        }
        Ok(Some(qargs.into_iter().flatten().collect_vec()))
    }

    /// Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
    /// for the target
    ///
    /// For globally defined variable width operations the tuple will be of the form
    /// ``(class, None)`` where class is the actual operation class that
    /// is globally defined.
    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, Option<Qargs>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Option<Qargs>)> = vec![];
        // Add all operations and dehash qargs.
        for (op, props_map) in self.gate_map.iter() {
            for qarg in props_map.keys() {
                let instruction_pair = (self._gate_name_map[op].clone_ref(py), qarg.clone());
                instruction_list.push(instruction_pair);
            }
        }
        // Return results.
        Ok(instruction_list)
    }
    /// Get the operation names in the target.
    #[getter]
    fn operation_names(&self) -> Vec<String> {
        self.gate_map.keys().cloned().collect()
    }

    /// Get the operation objects in the target.
    #[getter]
    fn operations(&self) -> Vec<PyObject> {
        return Vec::from_iter(self._gate_name_map.values().cloned());
    }

    /// Returns a sorted list of physical qubits.
    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        Vec::from_iter(0..self.num_qubits.unwrap_or_default())
    }

    // Magic methods:

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let result_list = PyList::empty_bound(py);
        result_list.append(self.description.clone())?;
        result_list.append(self.num_qubits)?;
        result_list.append(self.dt)?;
        result_list.append(self.granularity)?;
        result_list.append(self.min_length)?;
        result_list.append(self.pulse_alignment)?;
        result_list.append(self.acquire_alignment)?;
        result_list.append(self.qubit_properties.clone())?;
        result_list.append(self.concurrent_measurements.clone())?;
        result_list.append(
            self.gate_map
                .clone()
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        value
                            .into_iter()
                            .collect::<Vec<(Option<Qargs>, Option<BaseInstructionProperties>)>>(),
                    )
                })
                .collect::<GateMapState>()
                .into_py(py),
        )?;
        result_list.append(self._gate_name_map.clone())?;
        result_list.append(self.global_operations.clone())?;
        result_list.append(self.qarg_gate_map.clone().into_iter().collect_vec())?;
        result_list.append(self.non_global_basis.clone())?;
        result_list.append(self.non_global_strict_basis.clone())?;
        Ok(result_list.to_owned().unbind())
    }

    fn __setstate__(&mut self, state: Bound<PyList>) -> PyResult<()> {
        self.description = state.get_item(0)?.extract::<Option<String>>()?;
        self.num_qubits = state.get_item(1)?.extract::<Option<usize>>()?;
        self.dt = state.get_item(2)?.extract::<Option<f64>>()?;
        self.granularity = state.get_item(3)?.extract::<i32>()?;
        self.min_length = state.get_item(4)?.extract::<usize>()?;
        self.pulse_alignment = state.get_item(5)?.extract::<i32>()?;
        self.acquire_alignment = state.get_item(6)?.extract::<i32>()?;
        self.qubit_properties = state.get_item(7)?.extract::<Option<Vec<PyObject>>>()?;
        self.concurrent_measurements = state.get_item(8)?.extract::<Vec<Vec<usize>>>()?;
        self.gate_map = IndexMap::from_iter(
            state
                .get_item(9)?
                .extract::<GateMapState>()?
                .into_iter()
                .map(|(name, prop_map)| (name, IndexMap::from_iter(prop_map.into_iter()))),
        );
        self._gate_name_map = state
            .get_item(10)?
            .extract::<IndexMap<String, PyObject>>()?;
        self.global_operations = state
            .get_item(11)?
            .extract::<IndexMap<usize, HashSet<String>>>()?;
        self.qarg_gate_map = IndexMap::from_iter(
            state
                .get_item(12)?
                .extract::<Vec<(Option<Qargs>, Option<HashSet<String>>)>>()?,
        );
        self.non_global_basis = state.get_item(13)?.extract::<Option<Vec<String>>>()?;
        self.non_global_strict_basis = state.get_item(14)?.extract::<Option<Vec<String>>>()?;
        Ok(())
    }

    fn keys(&self) -> Vec<String> {
        self.gate_map.keys().cloned().collect()
    }

    fn values(&self) -> Vec<PropsMap> {
        self.gate_map.values().cloned().collect()
    }

    fn items(&self) -> Vec<(String, PropsMap)> {
        self.gate_map.clone().into_iter().collect_vec()
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BaseInstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
