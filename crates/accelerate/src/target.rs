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
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use hashbrown::HashSet;
use indexmap::IndexMap;
use itertools::Itertools;
use pyo3::{
    exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyTypeError},
    prelude::*,
    pyclass,
    types::{IntoPyDict, PyList, PyType},
};
use smallvec::{smallvec, IntoIter, SmallVec};

use crate::nlayout::PhysicalQubit;

use self::exceptions::{QiskitError, TranspilerError};

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

fn qubit_props_list_from_props(
    py: Python<'_>,
    properties: &Bound<PyAny>,
) -> PyResult<Vec<PyObject>> {
    let qiskit_backend_comp_module = py.import_bound("qiskit.providers.backend_compat")?;
    let qubit_props_list_funct =
        qiskit_backend_comp_module.getattr("qubit_props_list_from_props")?;
    let kwargs = [("properties", properties)].into_py_dict_bound(py);
    let props_list = qubit_props_list_funct.call((), Some(&kwargs))?;
    props_list.extract::<Vec<PyObject>>()
}

// Subclassable or Python Wrapping.
/**
 A representation of the properties of a gate implementation.

This class provides the optional properties that a backend can provide
about an instruction. These represent the set that the transpiler can
currently work with if present. However, if your backend provides additional
properties for instructions you should subclass this to add additional
custom attributes for those custom/additional properties by the backend.
*/
#[pyclass(subclass, module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
    #[pyo3(get)]
    _calibration: PyObject,
}

#[pymethods]
impl InstructionProperties {
    /**
    Create a new ``InstructionProperties`` object

    Args:
        duration (Option<f64>): The duration, in seconds, of the instruction on the
            specified set of qubits
        error (Option<f64>): The average error rate for the instruction on the specified
            set of qubits.
        calibration (Option<PyObject>): The pulse representation of the instruction.
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
            _calibration: py.None(),
        };
        if let Some(calibration) = calibration {
            let _ = instruction_prop.set_calibration(py, calibration);
        }
        instruction_prop
    }

    /**
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
    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> PyResult<PyObject> {
        if !&self._calibration.is_none(py) {
            return self._calibration.call_method0(py, "get_schedule");
        }
        Ok(py.None())
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
            self._calibration = new_entry.unbind();
        } else {
            self._calibration = calibration.unbind();
        }
        Ok(())
    }

    fn __getstate__(&self) -> PyResult<(Option<f64>, Option<f64>, Option<&PyObject>)> {
        Ok((self.duration, self.error, Some(&self._calibration)))
    }

    fn __setstate__(
        &mut self,
        py: Python<'_>,
        state: (Option<f64>, Option<f64>, Bound<PyAny>),
    ) -> PyResult<()> {
        self.duration = state.0;
        self.error = state.1;
        self.set_calibration(py, state.2)?;
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

        if !self.get_calibration(py)?.is_none(py) {
            output.push_str(
                format!(
                    "calibration={:?})",
                    self.get_calibration(py)?
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

#[pyclass]
struct CustomIter {
    iter: IntoIter<[PhysicalQubit; 4]>,
}

#[pymethods]
impl CustomIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        slf.iter.next().map(|next| next.to_object(slf.py()))
    }
}

// This struct allows quick transformation of qargs to tuple from and to python.
#[pyclass(sequence, module = "qiskit._accelerate.target")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Qargs {
    pub vec: SmallVec<[PhysicalQubit; 4]>,
}

#[pymethods]
impl Qargs {
    #[new]
    fn new(qargs: SmallVec<[PhysicalQubit; 4]>) -> Self {
        Qargs { vec: qargs }
    }

    fn __len__(&self) -> usize {
        self.vec.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<CustomIter>> {
        let iter = CustomIter {
            iter: slf.vec.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __hash__(slf: PyRef<'_, Self>) -> u64 {
        let mut hasher = DefaultHasher::new();
        slf.vec.hash(&mut hasher);
        hasher.finish()
    }

    fn __contains__(&self, obj: Bound<PyAny>) -> PyResult<bool> {
        if let Ok(obj) = obj.extract::<PhysicalQubit>() {
            Ok(self.vec.contains(&obj))
        } else {
            Ok(false)
        }
    }

    fn __getitem__(&self, obj: Bound<PyAny>) -> PyResult<PhysicalQubit> {
        if let Ok(index) = obj.extract::<usize>() {
            if let Some(item) = self.vec.get(index) {
                Ok(*item)
            } else {
                Err(PyKeyError::new_err(format!("Index {obj} is out of range.")))
            }
        } else {
            Err(PyTypeError::new_err(
                "Index type not supported.".to_string(),
            ))
        }
    }

    fn __getstate__(&self) -> PyResult<(QargsTuple,)> {
        Ok((self.vec.clone(),))
    }

    fn __setstate__(&mut self, py: Python<'_>, state: (PyObject,)) -> PyResult<()> {
        self.vec = state.0.extract::<QargsTuple>(py)?;
        Ok(())
    }

    fn __eq__(&self, other: Bound<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Qargs>() {
            self == &other
        } else if let Ok(other) = other.extract::<QargsTuple>() {
            self.vec == other
        } else {
            false
        }
    }
}

impl Default for Qargs {
    fn default() -> Self {
        Self { vec: smallvec![] }
    }
}

#[pyclass(mapping)]
#[derive(Debug, Clone)]
struct QargPropMap {
    map: GateMapValues,
}

#[pymethods]
impl QargPropMap {
    #[new]
    fn new(map: GateMapValues) -> Self {
        QargPropMap { map }
    }

    fn __contains__(&self, key: Bound<PyAny>) -> bool {
        if let Ok(key) = key.extract::<QargsTuple>() {
            let qarg = Some(Qargs::new(key));
            self.map.contains_key(&qarg)
        } else {
            false
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<PyAny>) -> PyResult<PyObject> {
        let key = if let Ok(qargs) = key.extract::<QargsTuple>() {
            Ok(Some(Qargs::new(qargs)))
        } else if let Ok(qargs) = key.extract::<Qargs>() {
            Ok(Some(qargs))
        } else {
            Err(PyKeyError::new_err(format!(
                "Key {:#?} not in target.",
                key
            )))
        }?;
        if let Some(item) = self.map.get(&key) {
            Ok(item.to_object(py))
        } else {
            Err(PyKeyError::new_err(format!(
                "Key {:#?} not in target.",
                key.unwrap_or_default().vec
            )))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: Bound<PyAny>, default: Option<Bound<PyAny>>) -> PyObject {
        match self.__getitem__(py, key) {
            Ok(value) => value,
            Err(_) => match default {
                Some(value) => value.into(),
                None => py.None(),
            },
        }
    }

    fn keys(&self) -> HashSet<Option<Qargs>> {
        self.map.keys().cloned().collect()
    }

    fn values(&self) -> Vec<Option<Py<InstructionProperties>>> {
        self.map.clone().into_values().collect_vec()
    }

    fn items(&self) -> Vec<(Option<Qargs>, Option<Py<InstructionProperties>>)> {
        self.map.clone().into_iter().collect_vec()
    }
}

// Custom types
type QargsTuple = SmallVec<[PhysicalQubit; 4]>;
type GateMapType = IndexMap<String, QargPropMap>;
type GateMapValues = IndexMap<Option<Qargs>, Option<Py<InstructionProperties>>>;
type ErrorDictType<'a> = IndexMap<String, IndexMap<QargsTuple, Bound<'a, PyAny>>>;

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
    pub qubit_properties: Vec<PyObject>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Vec<Vec<usize>>,
    // Maybe convert PyObjects into rust representations of Instruction and Data
    gate_map: GateMapType,
    gate_name_map: IndexMap<String, PyObject>,
    global_operations: IndexMap<usize, HashSet<String>>,
    qarg_gate_map: IndexMap<Option<Qargs>, Option<HashSet<String>>>,
    instruction_durations: Option<PyObject>,
    instruction_schedule_map: Option<PyObject>,
    #[pyo3(get, set)]
    coupling_graph: Option<PyObject>,
    non_global_strict_basis: Option<Vec<String>>,
    non_global_basis: Option<Vec<String>>,
}

#[pymethods]
impl Target {
    /**
    Create a new ``Target`` object

    Args:
        description (str): An optional string to describe the Target.
        num_qubits (int): An optional int to specify the number of qubits
            the backend target has. If not set it will be implicitly set
            based on the qargs when :meth:`~qiskit.Target.add_instruction`
            is called. Note this must be set if the backend target is for a
            noiseless simulator that doesn't have constraints on the
            instructions so the transpiler knows how many qubits are
            available.
        dt (float): The system time resolution of input signals in seconds
        granularity (int): An integer value representing minimum pulse gate
            resolution in units of ``dt``. A user-defined pulse gate should
            have duration of a multiple of this granularity value.
        min_length (int): An integer value representing minimum pulse gate
            length in units of ``dt``. A user-defined pulse gate should be
            longer than this length.
        pulse_alignment (int): An integer value representing a time
            resolution of gate instruction starting time. Gate instruction
            should start at time which is a multiple of the alignment
            value.
        acquire_alignment (int): An integer value representing a time
            resolution of measure instruction starting time. Measure
            instruction should start at time which is a multiple of the
            alignment value.
        qubit_properties (list): A list of :class:`~.QubitProperties`
            objects defining the characteristics of each qubit on the
            target device. If specified the length of this list must match
            the number of qubits in the target, where the index in the list
            matches the qubit number the properties are defined for. If some
            qubits don't have properties available you can set that entry to
            ``None``
        concurrent_measurements(list): A list of sets of qubits that must be
            measured together. This must be provided
            as a nested list like ``[[0, 1], [2, 3, 4]]``.
    Raises:
        ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
            defined and the value of ``num_qubits`` differs from the length of
            ``qubit_properties``.
     */
    #[new]
    #[pyo3(text_signature = "(/,\
        description=None,\
        num_qubits=0,\
        dt=None,\
        granularity=1,\
        min_length=1,\
        pulse_alignment=1,\
        acquire_alignment=1,\
        qubit_properties=None,\
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

    /**
    Add a new instruction to the :class:`~qiskit.transpiler.Target`

    As ``Target`` objects are strictly additive this is the primary method
    for modifying a ``Target``. Typically, you will use this to fully populate
    a ``Target`` before using it in :class:`~qiskit.providers.BackendV2`. For
    example::

        from qiskit.circuit.library import CXGate
        from qiskit.transpiler import Target, InstructionProperties

        target = Target()
        cx_properties = {
            (0, 1): None,
            (1, 0): None,
            (0, 2): None,
            (2, 0): None,
            (0, 3): None,
            (2, 3): None,
            (3, 0): None,
            (3, 2): None
        }
        target.add_instruction(CXGate(), cx_properties)

    Will add a :class:`~qiskit.circuit.library.CXGate` to the target with no
    properties (duration, error, etc) with the coupling edge list:
    ``(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (2, 3), (3, 0), (3, 2)``. If
    there are properties available for the instruction you can replace the
    ``None`` value in the properties dictionary with an
    :class:`~qiskit.transpiler.InstructionProperties` object. This pattern
    is repeated for each :class:`~qiskit.circuit.Instruction` the target
    supports.

    Args:
        instruction (Union[qiskit.circuit.Instruction, Type[qiskit.circuit.Instruction]]):
            The operation object to add to the map. If it's parameterized any value
            of the parameter can be set. Optionally for variable width
            instructions (such as control flow operations such as :class:`~.ForLoop` or
            :class:`~MCXGate`) you can specify the class. If the class is specified than the
            ``name`` argument must be specified. When a class is used the gate is treated as global
            and not having any properties set.
        properties (dict): A dictionary of qarg entries to an
            :class:`~qiskit.transpiler.InstructionProperties` object for that
            instruction implementation on the backend. Properties are optional
            for any instruction implementation, if there are no
            :class:`~qiskit.transpiler.InstructionProperties` available for the
            backend the value can be None. If there are no constraints on the
            instruction (as in a noiseless/ideal simulation) this can be set to
            ``{None, None}`` which will indicate it runs on all qubits (or all
            available permutations of qubits for multi-qubit gates). The first
            ``None`` indicates it applies to all qubits and the second ``None``
            indicates there are no
            :class:`~qiskit.transpiler.InstructionProperties` for the
            instruction. By default, if properties is not set it is equivalent to
            passing ``{None: None}``.
        name (str): An optional name to use for identifying the instruction. If not
            specified the :attr:`~qiskit.circuit.Instruction.name` attribute
            of ``gate`` will be used. All gates in the ``Target`` need unique
            names. Backends can differentiate between different
            parameterization of a single gate by providing a unique name for
            each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
            documentation for the :class:`~qiskit.transpiler.Target` class).
    Raises:
        AttributeError: If gate is already in map
        TranspilerError: If an operation class is passed in for ``instruction`` and no name
            is specified or ``properties`` is set.
     */
    #[pyo3(signature = (instruction, /, properties=None, name=None))]
    fn add_instruction(
        &mut self,
        py: Python<'_>,
        instruction: &Bound<PyAny>,
        properties: Option<IndexMap<Option<QargsTuple>, Option<Py<InstructionProperties>>>>,
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
        let mut qargs_val: IndexMap<Option<Qargs>, Option<Py<InstructionProperties>>> =
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
                    .or_insert(HashSet::from_iter([instruction_name.clone()]));
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
                    qarg_obj = Some(Qargs::new(qarg.clone()))
                }
                qargs_val.insert(qarg_obj.to_owned(), properties[qarg].clone());
                self.qarg_gate_map
                    .entry(qarg_obj)
                    .and_modify(|e| {
                        if let Some(e) = e {
                            e.insert(instruction_name.clone());
                        }
                    })
                    .or_insert(Some(HashSet::from([instruction_name.clone()])));
            }
        }
        self.gate_map
            .insert(instruction_name, QargPropMap::new(qargs_val));
        self.coupling_graph = None;
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        self.non_global_basis = None;
        self.non_global_strict_basis = None;
        Ok(())
    }

    /**
    Update the property object for an instruction qarg pair already in the Target

    Args:
        instruction (str): The instruction name to update
        qargs (tuple): The qargs to update the properties of
        properties (InstructionProperties): The properties to set for this instruction
    Raises:
        KeyError: If ``instruction`` or ``qarg`` are not in the target
     */
    #[pyo3(text_signature = "(instruction, qargs, properties, /,)")]
    fn update_instruction_properties(
        &mut self,
        _py: Python<'_>,
        instruction: String,
        qargs: Option<QargsTuple>,
        properties: Option<Py<InstructionProperties>>,
    ) -> PyResult<()> {
        if !self.gate_map.contains_key(&instruction) {
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        let qargs = qargs.map(Qargs::new);
        if !(self.gate_map[&instruction].map.contains_key(&qargs)) {
            return Err(PyKeyError::new_err(format!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs.unwrap_or_default().vec,
                &instruction
            )));
        }
        if let Some(q_vals) = self.gate_map.get_mut(&instruction) {
            if let Some(q_vals) = q_vals.map.get_mut(&qargs) {
                *q_vals = properties;
            }
        }
        self.instruction_durations = None;
        self.instruction_schedule_map = None;
        Ok(())
    }

    /**
    Update the target from an instruction schedule map.

    If the input instruction schedule map contains new instructions not in
    the target they will be added. However, if it contains additional qargs
    for an existing instruction in the target it will error.

    Args:
        inst_map (InstructionScheduleMap): The instruction
        inst_name_map (dict): An optional dictionary that maps any
            instruction name in ``inst_map`` to an instruction object.
            If not provided, instruction is pulled from the standard Qiskit gates,
            and finally custom gate instance is created with schedule name.
        error_dict (dict): A dictionary of errors of the form::

            {gate_name: {qarg: error}}

        for example::

            {'rx': {(0, ): 1.4e-4, (1, ): 1.2e-4}}

        For each entry in the ``inst_map`` if ``error_dict`` is defined
        a when updating the ``Target`` the error value will be pulled from
        this dictionary. If one is not found in ``error_dict`` then
        ``None`` will be used.
     */
    #[pyo3(text_signature = "(inst_map, /, inst_name_map=None, error_dict=None)")]
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
            let mut out_prop: IndexMap<Option<QargsTuple>, Option<Py<InstructionProperties>>> =
                IndexMap::new();
            let inst_map_qubit_instruction_for_name =
                inst_map.call_method1("qubits_with_instruction", (&inst_name,))?;
            let inst_map_qubit_instruction_for_name =
                inst_map_qubit_instruction_for_name.downcast::<PyList>()?;
            for qargs in inst_map_qubit_instruction_for_name {
                let qargs_: QargsTuple = if let Ok(qargs_to_tuple) = qargs.extract::<QargsTuple>() {
                    qargs_to_tuple
                } else {
                    smallvec![qargs.extract::<PhysicalQubit>()?]
                };
                let opt_qargs = Some(Qargs::new(qargs_.clone()));
                let mut props: Option<Py<InstructionProperties>> =
                    if let Some(prop_value) = self.gate_map.get(&inst_name) {
                        if let Some(prop) = prop_value.map.get(&opt_qargs) {
                            prop.clone()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                let entry = get_calibration
                    .call1((&inst_name, qargs_.iter().cloned().collect::<QargsTuple>()))?;
                let entry_comparison: bool = if let Some(props) = &props {
                    !entry.eq(&props.getattr(py, "_calibration")?)?
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
                    props = Some(Py::new(
                        py,
                        InstructionProperties::new(py, duration, None, Some(entry)),
                    )?);
                } else if props.is_none() {
                    continue;
                }

                if let Some(error_dict) = error_dict.as_ref() {
                    if let Some(error_dict_name) = error_dict.get(&inst_name) {
                        if let (Some(error_prop), Some(props_)) =
                            (error_dict_name.get(&qargs_), props.as_mut())
                        {
                            props_.setattr(py, "error", Some(error_prop.extract::<f64>()?))?;
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
                        Option<QargsTuple>,
                        Option<Py<InstructionProperties>>,
                    > = IndexMap::new();
                    for (qargs, prop) in out_prop.iter() {
                        if qargs.as_ref().unwrap_or(&smallvec![]).len()
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
                    let mut param_names: HashSet<Vec<String>> = HashSet::new();
                    let inst_map_qubit_instruction_for_name =
                        inst_map.call_method1("qubits_with_instruction", (&inst_name,))?;
                    let inst_map_qubit_instruction_for_name =
                        inst_map_qubit_instruction_for_name.downcast::<PyList>()?;
                    for qargs in inst_map_qubit_instruction_for_name {
                        let qargs_ = if let Ok(qargs_ext) = qargs.extract::<QargsTuple>() {
                            qargs_ext
                        } else {
                            smallvec![qargs.extract::<PhysicalQubit>()?]
                        };
                        qlen.insert(qargs_.len());
                        let cal = if let Some(Some(prop)) = out_prop.get(&Some(qargs_)) {
                            Some(prop.getattr(py, "_calibration")?)
                        } else {
                            None
                        };
                        if let Some(cal) = cal {
                            let params = cal
                                .call_method0(py, "get_signature")?
                                .getattr(py, "parameters")?
                                .call_method0(py, "keys")?
                                .extract::<Vec<String>>(py)?;
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
                                qlen.iter().collect::<Vec<&usize>>(),
                                param_names.iter().collect::<Vec<&Vec<String>>>()
                            )));
                        }
                        let gate_class = py.import_bound("qiskit.circuit.gate")?.getattr("Gate")?;
                        let parameter_class = py
                            .import_bound("qiskit.circuit.parameter")?
                            .getattr("Parameter")?;
                        let params =
                            parameter_class.call1((param_names.iter().next().to_object(py),))?;
                        let kwargs = [
                            ("name", inst_name.as_str().into_py(py)),
                            ("num_qubits", qlen.iter().next().to_object(py)),
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
                for (qargs, prop) in out_prop.into_iter() {
                    if let Some(gate_inst) = self.gate_map.get(&inst_name) {
                        if !gate_inst
                            .map
                            .contains_key(&Some(Qargs::new(qargs.to_owned().unwrap_or_default())))
                        {
                            continue;
                        }
                    }
                    self.update_instruction_properties(py, inst_name.to_owned(), qargs, prop)?;
                }
            }
        }
        Ok(())
    }

    /**
    Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
    instructions in the target with a pulse schedule defined.

    Returns:
        InstructionScheduleMap: The instruction schedule map for the
        instructions in this target with a pulse schedule defined.
     */
    #[pyo3(text_signature = "(/)")]
    fn instruction_schedule_map(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(schedule_map) = self.instruction_schedule_map.as_ref() {
            return Ok(schedule_map.to_owned());
        }
        let inst_sched_map_module = py.import_bound("qiskit.pulse.instruction_schedule_map")?;
        let inst_sched_map_class = inst_sched_map_module.getattr("InstructionScheduleMap")?;
        let out_inst_schedule_map = inst_sched_map_class.call0()?;
        for (instruction, qargs) in self.gate_map.iter() {
            for (qarg, properties) in qargs.map.iter() {
                // Directly getting calibration entry to invoke .get_schedule().
                // This keeps PulseQobjDef unparsed.
                if let Some(properties) = properties {
                    let cal_entry = &properties.getattr(py, "_calibration")?;
                    if !cal_entry.is_none(py) {
                        let _ = out_inst_schedule_map
                            .call_method1("_add", (instruction, qarg.to_owned(), cal_entry));
                    }
                }
            }
        }
        self.instruction_schedule_map = Some(out_inst_schedule_map.clone().unbind());
        Ok(out_inst_schedule_map.unbind())
    }

    /**
    Get the qargs for a given operation name

    Args:
        operation (str): The operation name to get qargs for
    Returns:
        set: The set of qargs the gate instance applies to.
    */
    #[pyo3(text_signature = "(operation, /,)")]
    fn qargs_for_operation_name(&self, operation: String) -> PyResult<Option<Vec<Option<Qargs>>>> {
        if let Some(gate_map_oper) = self.gate_map.get(&operation) {
            if gate_map_oper.map.contains_key(&None) {
                return Ok(None);
            }
            let qargs: Vec<Option<Qargs>> = gate_map_oper.map.keys().cloned().collect();
            Ok(Some(qargs))
        } else {
            Err(PyKeyError::new_err(format!(
                "Operation: {operation} not in Target."
            )))
        }
    }

    /**
    Get an InstructionDurations object from the target

    Returns:
        InstructionDurations: The instruction duration represented in the
            target
    */
    #[pyo3(text_signature = "(/,)")]
    fn durations(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.instruction_durations.is_some() {
            return Ok(self.instruction_durations.to_owned());
        }
        let mut out_durations: Vec<(&String, Qargs, f64, &str)> = vec![];
        for (instruction, props_map) in self.gate_map.iter() {
            for (qarg, properties) in props_map.map.iter() {
                if let Some(properties) = properties {
                    if let Some(duration) = properties.getattr(py, "duration")?.extract(py)? {
                        out_durations.push((
                            instruction,
                            qarg.to_owned().unwrap_or_default(),
                            duration,
                            "s",
                        ))
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

    /**
    Get an :class:`~qiskit.transpiler.TimingConstraints` object from the target

    Returns:
        TimingConstraints: The timing constraints represented in the ``Target``
    */
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

    /**
    Get the operation class object for a given name

    Args:
        instruction (str): The instruction name to get the
            :class:`~qiskit.circuit.Instruction` instance for
    Returns:
        qiskit.circuit.Instruction: The Instruction instance corresponding to the
        name. This also can also be the class for globally defined variable with
        operations.
    */
    #[pyo3(text_signature = "(instruction, /)")]
    fn operation_from_name(&self, py: Python<'_>, instruction: String) -> PyResult<PyObject> {
        if let Some(gate_obj) = self.gate_name_map.get(&instruction) {
            Ok(gate_obj.to_object(py))
        } else {
            Err(PyKeyError::new_err(format!(
                "Instruction {:?} not in target",
                instruction
            )))
        }
    }

    /**
    Get the operation class object for a specified qargs tuple

    Args:
        qargs (tuple): A qargs tuple of the qubits to get the gates that apply
            to it. For example, ``(0,)`` will return the set of all
            instructions that apply to qubit 0. If set to ``None`` this will
            return any globally defined operations in the target.
    Returns:
        list: The list of :class:`~qiskit.circuit.Instruction` instances
        that apply to the specified qarg. This may also be a class if
        a variable width operation is globally defined.

    Raises:
        KeyError: If qargs is not in target
    */
    #[pyo3(text_signature = "(/, qargs=None)")]
    fn operations_for_qargs(&self, py: Python<'_>, qargs: Option<Qargs>) -> PyResult<Py<PyList>> {
        let res = PyList::empty_bound(py);
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .vec
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
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
                res.append(&self.gate_name_map[x])?;
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

    /**
    Get the operation names for a specified qargs tuple

    Args:
        qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
            to it. For example, ``(0,)`` will return the set of all
            instructions that apply to qubit 0. If set to ``None`` this will
            return the names for any globally defined operations in the target.
    Returns:
        set: The set of operation names that apply to the specified ``qargs``.

    Raises:
        KeyError: If ``qargs`` is not in target
    */
    #[pyo3(text_signature = "(/, qargs=None)")]
    fn operation_names_for_qargs(
        &self,
        py: Python<'_>,
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
                .vec
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
            {
                return Err(PyKeyError::new_err(format!("{:?}", qargs)));
            }
        }
        if let Some(Some(qarg_gate_map_arg)) = self.qarg_gate_map.get(&qargs).as_ref() {
            res.extend(qarg_gate_map_arg);
        }
        for (name, op) in self.gate_name_map.iter() {
            if isclass(py, op.bind(py))? {
                res.insert(name);
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(global_gates) = self.global_operations.get(&qargs.vec.len()) {
                for gate_name_ref in global_gates {
                    res.insert(gate_name_ref);
                }
            }
        }
        if res.is_empty() {
            return Err(PyKeyError::new_err(format!("{:?} not in target", qargs)));
        }
        Ok(res)
    }

    /**
    Return whether the instruction (operation + qubits) is supported by the target

    Args:
        operation_name (str): The name of the operation for the instruction. Either
            this or ``operation_class`` must be specified, if both are specified
            ``operation_class`` will take priority and this argument will be ignored.
        qargs (tuple): The tuple of qubit indices for the instruction. If this is
            not specified then this method will return ``True`` if the specified
            operation is supported on any qubits. The typical application will
            always have this set (otherwise it's the same as just checking if the
            target contains the operation). Normally you would not set this argument
            if you wanted to check more generally that the target supports an operation
            with the ``parameters`` on any qubits.
        operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
            the target supports a particular operation by class rather
            than by name. This lookup is more expensive as it needs to
            iterate over all operations in the target instead of just a
            single lookup. If this is specified it will supersede the
            ``operation_name`` argument. The typical use case for this
            operation is to check whether a specific variant of an operation
            is supported on the backend. For example, if you wanted to
            check whether a :class:`~.RXGate` was supported on a specific
            qubit with a fixed angle. That fixed angle variant will
            typically have a name different from the object's
            :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
            This can be used to check if any instances of the class are
            available in such a case.
        parameters (list): A list of parameters to check if the target
            supports them on the specified qubits. If the instruction
            supports the parameter values specified in the list on the
            operation and qargs specified this will return ``True`` but
            if the parameters are not supported on the specified
            instruction it will return ``False``. If this argument is not
            specified this method will return ``True`` if the instruction
            is supported independent of the instruction parameters. If
            specified with any :class:`~.Parameter` objects in the list,
            that entry will be treated as supporting any value, however parameter names
            will not be checked (for example if an operation in the target
            is listed as parameterized with ``"theta"`` and ``"phi"`` is
            passed into this function that will return ``True``). For
            example, if called with::

                parameters = [Parameter("theta")]
                target.instruction_supported("rx", (0,), parameters=parameters)

            will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
            that will accept any parameter. If you need to check for a fixed numeric
            value parameter this argument is typically paired with the ``operation_class``
            argument. For example::

                target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])

            will return ``True`` if an RXGate(pi/4) exists on qubit 0.

    Returns:
        bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.
    */
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
        let parameter_module = py.import_bound("qiskit.circuit.parameter")?;
        let parameter_class = parameter_module.getattr("Parameter")?;
        let parameter_class = parameter_class.downcast::<PyType>()?;

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
                if param.eq(&param_at_index)? && !param_at_index.is_instance(parameter_class)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };

        if self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(operation_class) = operation_class {
            for (op_name, obj) in self.gate_name_map.iter() {
                if isclass(py, obj.bind(py))? {
                    if !operation_class.eq(obj)? {
                        continue;
                    }
                    // If no qargs operation class is supported
                    if let Some(_qargs) = &qargs {
                        let qarg_set: HashSet<PhysicalQubit> = _qargs.vec.iter().cloned().collect();
                        // If qargs set then validate no duplicates and all indices are valid on device
                        if _qargs
                            .vec
                            .iter()
                            .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                            && qarg_set.len() == _qargs.vec.len()
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
                            if gate_map_name.map.contains_key(&qargs) {
                                return Ok(true);
                            }
                            if gate_map_name.map.contains_key(&None) {
                                let qubit_comparison = self.gate_name_map[op_name]
                                    .getattr(py, "num_qubits")?
                                    .extract::<usize>(py)?;
                                return Ok(qubit_comparison == _qargs.vec.len()
                                    && _qargs
                                        .vec
                                        .iter()
                                        .all(|x| x.index() < self.num_qubits.unwrap_or_default()));
                            }
                        } else {
                            let qubit_comparison = self.gate_name_map[op_name]
                                .getattr(py, "num_qubits")?
                                .extract::<usize>(py)?;
                            return Ok(qubit_comparison == _qargs.vec.len()
                                && _qargs
                                    .vec
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
                    let obj = self.gate_name_map[operation_names].to_owned();
                    if isclass(py, obj.bind(py))? {
                        if let Some(_qargs) = qargs {
                            let qarg_set: HashSet<PhysicalQubit> =
                                _qargs.vec.iter().cloned().collect();
                            if _qargs
                                .vec
                                .iter()
                                .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                && qarg_set.len() == _qargs.vec.len()
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
                    let qarg_set: HashSet<PhysicalQubit> = _qargs.vec.iter().cloned().collect();
                    if self.gate_map.contains_key(operation_names) {
                        let gate_map_name = &self.gate_map[operation_names];
                        if gate_map_name.map.contains_key(&qargs) {
                            return Ok(true);
                        }
                        if gate_map_name.map.contains_key(&None) {
                            let obj = &self.gate_name_map[operation_names];
                            if isclass(py, obj.bind(py))? {
                                if qargs.is_none()
                                    || _qargs.vec.iter().all(|qarg| {
                                        qarg.index() <= self.num_qubits.unwrap_or_default()
                                    }) && qarg_set.len() == _qargs.vec.len()
                                {
                                    return Ok(true);
                                } else {
                                    return Ok(false);
                                }
                            } else {
                                let qubit_comparison = self.gate_name_map[operation_names]
                                    .getattr(py, "num_qubits")?
                                    .extract::<usize>(py)?;
                                return Ok(qubit_comparison == _qargs.vec.len()
                                    && _qargs.vec.iter().all(|qarg| {
                                        qarg.index() < self.num_qubits.unwrap_or_default()
                                    }));
                            }
                        }
                    } else {
                        // Duplicate case is if it contains none
                        let obj = &self.gate_name_map[operation_names];
                        if isclass(py, obj.bind(py))? {
                            if qargs.is_none()
                                || _qargs
                                    .vec
                                    .iter()
                                    .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                    && qarg_set.len() == _qargs.vec.len()
                            {
                                return Ok(true);
                            } else {
                                return Ok(false);
                            }
                        } else {
                            let qubit_comparison = self.gate_name_map[operation_names]
                                .getattr(py, "num_qubits")?
                                .extract::<usize>(py)?;
                            return Ok(qubit_comparison == _qargs.vec.len()
                                && _qargs.vec.iter().all(|qarg| {
                                    qarg.index() < self.num_qubits.unwrap_or_default()
                                }));
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /**
    Return whether the instruction (operation + qubits) defines a calibration.

    Args:
        operation_name: The name of the operation for the instruction.
        qargs: The tuple of qubit indices for the instruction.

    Returns:
        Returns ``True`` if the calibration is supported and ``False`` if it isn't.
    */
    #[pyo3(text_signature = "( /, operation_name: str, qargs: tuple[int, ...],)")]
    fn has_calibration(
        &self,
        py: Python<'_>,
        operation_name: String,
        qargs: Qargs,
    ) -> PyResult<bool> {
        if !self.gate_map.contains_key(&operation_name) {
            return Ok(false);
        }
        if self.gate_map.contains_key(&operation_name) {
            let gate_map_qarg = &self.gate_map[&operation_name];
            if let Some(oper_qarg) = &gate_map_qarg.map[&Some(qargs)] {
                return Ok(!oper_qarg.getattr(py, "_calibration")?.is_none(py));
            } else {
                return Ok(false);
            }
        }
        Ok(false)
    }

    /**
    Get calibrated pulse schedule for the instruction.

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
    #[pyo3(
        text_signature = "( /, operation_name: str, qargs: tuple[int, ...], *args: ParameterValueType, **kwargs: ParameterValueType,)"
    )]
    fn get_calibration(
        &self,
        py: Python<'_>,
        operation_name: String,
        qargs: Qargs,
    ) -> PyResult<PyObject> {
        if !self.has_calibration(py, operation_name.clone(), qargs.clone())? {
            return Err(PyKeyError::new_err(format!(
                "Calibration of instruction {:?} for qubit {:?} is not defined.",
                operation_name, qargs.vec
            )));
        }

        self.gate_map[&operation_name].map[&Some(qargs)]
            .as_ref()
            .unwrap()
            .getattr(py, "_calibration")
    }

    /**
    Get the instruction properties for a specific instruction tuple

    This method is to be used in conjunction with the
    :attr:`~qiskit.transpiler.Target.instructions` attribute of a
    :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
    get the instruction properties for an element of
    :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
    However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
    directly it is likely more efficient to access the target directly via the name
    and qubits to get the instruction properties. For example, if
    :attr:`~qiskit.transpiler.Target.instructions` returned::

        [(XGate(), (0,)), (XGate(), (1,))]

    you could get the properties of the ``XGate`` on qubit 1 with::

        props = target.instruction_properties(1)

    but just accessing it directly via the name would be more efficient::

        props = target['x'][(1,)]

    (assuming the ``XGate``'s canonical name in the target is ``'x'``)
    This is especially true for larger targets as this will scale worse with the number
    of instruction tuples in a target.

    Args:
        index (int): The index of the instruction tuple from the
            :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
            if you want the properties from the third element in
            :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
    Returns:
        InstructionProperties: The instruction properties for the specified instruction tuple
    */
    #[pyo3(text_signature = "(/, index: int)")]
    fn instruction_properties(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        let mut instruction_properties: Vec<PyObject> = vec![];
        for operation in self.gate_map.keys() {
            if self.gate_map.contains_key(operation) {
                let gate_map_oper = &self.gate_map[operation];
                for (_, inst_props) in gate_map_oper.map.iter() {
                    instruction_properties.push(inst_props.to_object(py))
                }
            }
        }
        if !((0..instruction_properties.len()).contains(&index)) {
            return Err(PyIndexError::new_err(format!(
                "Index: {:?} is out of range.",
                index
            )));
        }
        Ok(instruction_properties[index].to_object(py))
    }

    /**
    Return the non-global operation names for the target

    The non-global operations are those in the target which don't apply
    on all qubits (for single qubit operations) or all multi-qubit qargs
    (for multi-qubit operations).

    Args:
        strict_direction (bool): If set to ``True`` the multi-qubit
            operations considered as non-global respect the strict
            direction (or order of qubits in the qargs is significant). For
            example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
            defined over ``(1, 0)`` by default neither would be considered
            non-global, but if ``strict_direction`` is set ``True`` both
            ``cx`` and ``ecr`` would be returned.

    Returns:
        List[str]: A list of operation names for operations that aren't global in this target
    */
    #[pyo3(signature = (/, strict_direction=false,), text_signature = "(/, strict_direction=false)")]
    fn get_non_global_operation_names(&mut self, strict_direction: bool) -> PyResult<Vec<String>> {
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
            for qarg_key in self.qarg_gate_map.keys().cloned() {
                if let Some(qarg_key_) = &qarg_key {
                    if qarg_key_.vec.len() != 1 {
                        let mut vec = qarg_key_.clone().vec;
                        vec.sort();
                        let qarg_key = Some(Qargs { vec });
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
                    .unwrap_or(&Qargs { vec: smallvec![] })
                    .vec
                    .len()
                    == 1
            {
                continue;
            }
            *size_dict
                .entry(qarg.to_owned().unwrap_or_default().vec.len())
                .or_insert(0) += 1;
        }
        for (inst, qargs_props) in self.gate_map.iter() {
            let mut qarg_len = qargs_props.map.len();
            let qarg_sample = qargs_props.map.keys().next();
            if let Some(qarg_sample) = qarg_sample {
                if !strict_direction {
                    let mut qarg_set = HashSet::new();
                    for qarg in qargs_props.keys() {
                        let mut qarg_set_vec: Qargs = Qargs { vec: smallvec![] };
                        if let Some(qarg) = qarg {
                            let mut to_vec = qarg.vec.to_owned();
                            to_vec.sort();
                            qarg_set_vec = Qargs { vec: to_vec };
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
    fn qargs(&self) -> PyResult<Option<HashSet<Option<Qargs>>>> {
        let qargs: HashSet<Option<Qargs>> = self.qarg_gate_map.keys().cloned().collect();
        // Modify logic to account for the case of {None}
        let next_entry = qargs.iter().flatten().next();
        if qargs.len() == 1 && (qargs.iter().next().is_none() || next_entry.is_none()) {
            return Ok(None);
        }
        Ok(Some(qargs))
    }

    /**
    Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
    for the target

    For globally defined variable width operations the tuple will be of the form
    ``(class, None)`` where class is the actual operation class that
    is globally defined.
    */
    #[getter]
    fn instructions(&self) -> PyResult<Vec<(PyObject, Option<Qargs>)>> {
        // Get list of instructions.
        let mut instruction_list: Vec<(PyObject, Option<Qargs>)> = vec![];
        // Add all operations and dehash qargs.
        for op in self.gate_map.keys() {
            if self.gate_map.contains_key(op) {
                let gate_map_op = &self.gate_map[op];
                for qarg in gate_map_op.keys() {
                    let instruction_pair = (self.gate_name_map[op].clone(), qarg.clone());
                    instruction_list.push(instruction_pair);
                }
            }
        }
        // Return results.
        Ok(instruction_list)
    }
    /// Get the operation names in the target.
    #[getter]
    fn operation_names(&self) -> HashSet<String> {
        return HashSet::from_iter(self.gate_map.keys().cloned());
    }
    /// Get the operation names in the target.
    #[getter]
    fn operations(&self) -> Vec<PyObject> {
        return Vec::from_iter(self.gate_name_map.values().cloned());
    }

    /// Returns a sorted list of physical qubits.
    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        Vec::from_iter(0..self.num_qubits.unwrap_or_default())
    }

    /**
    Create a target object from the individual global configuration

    Prior to the creation of the :class:`~.Target` class, the constraints
    of a backend were represented by a collection of different objects
    which combined represent a subset of the information contained in
    the :class:`~.Target`. This function provides a simple interface
    to convert those separate objects to a :class:`~.Target`.

    This constructor will use the input from ``basis_gates``, ``num_qubits``,
    and ``coupling_map`` to build a base model of the backend and the
    ``instruction_durations``, ``backend_properties``, and ``inst_map`` inputs
    are then queried (in that order) based on that model to look up the properties
    of each instruction and qubit. If there is an inconsistency between the inputs
    any extra or conflicting information present in ``instruction_durations``,
    ``backend_properties``, or ``inst_map`` will be ignored.

    Args:
        basis_gates: The list of basis gate names for the backend. For the
            target to be created these names must either be in the output
            from :func:`~.get_standard_gate_name_mapping` or present in the
            specified ``custom_name_mapping`` argument.
        num_qubits: The number of qubits supported on the backend.
        coupling_map: The coupling map representing connectivity constraints
            on the backend. If specified all gates from ``basis_gates`` will
            be supported on all qubits (or pairs of qubits).
        inst_map: The instruction schedule map representing the pulse
            :class:`~.Schedule` definitions for each instruction. If this
            is specified ``coupling_map`` must be specified. The
            ``coupling_map`` is used as the source of truth for connectivity
            and if ``inst_map`` is used the schedule is looked up based
            on the instructions from the pair of ``basis_gates`` and
            ``coupling_map``. If you want to define a custom gate for
            a particular qubit or qubit pair, you can manually build :class:`.Target`.
        backend_properties: The :class:`~.BackendProperties` object which is
            used for instruction properties and qubit properties.
            If specified and instruction properties are intended to be used
            then the ``coupling_map`` argument must be specified. This is
            only used to lookup error rates and durations (unless
            ``instruction_durations`` is specified which would take
            precedence) for instructions specified via ``coupling_map`` and
            ``basis_gates``.
        instruction_durations: Optional instruction durations for instructions. If specified
            it will take priority for setting the ``duration`` field in the
            :class:`~InstructionProperties` objects for the instructions in the target.
        concurrent_measurements(list): A list of sets of qubits that must be
            measured together. This must be provided
            as a nested list like ``[[0, 1], [2, 3, 4]]``.
        dt: The system time resolution of input signals in seconds
        timing_constraints: Optional timing constraints to include in the
            :class:`~.Target`
        custom_name_mapping: An optional dictionary that maps custom gate/operation names in
            ``basis_gates`` to an :class:`~.Operation` object representing that
            gate/operation. By default, most standard gates names are mapped to the
            standard gate object from :mod:`qiskit.circuit.library` this only needs
            to be specified if the input ``basis_gates`` defines gates in names outside
            that set.

    Returns:
        Target: the target built from the input configuration

    Raises:
        TranspilerError: If the input basis gates contain > 2 qubits and ``coupling_map`` is
        specified.
        KeyError: If no mapping is available for a specified ``basis_gate``.
    */
    #[classmethod]
    fn from_configuration(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
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
            qubit_properties = Some(qubit_props_list_from_props(py, backend_properties)?);
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
        let mut name_mapping = get_standard_gate_name_mapping(py)?;
        if let Some(custom_name_mapping) = custom_name_mapping {
            for (key, value) in custom_name_mapping.into_iter() {
                name_mapping.insert(key, value);
            }
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
                if let Some(gate_obj) = name_mapping.get(&gate) {
                    let gate_obj_num_qubits = gate_obj.getattr("num_qubits")?.extract::<usize>()?;
                    if gate_obj_num_qubits == 1 {
                        one_qubit_gates.push(gate);
                    } else if gate_obj_num_qubits == 2 {
                        two_qubit_gates.push(gate);
                    } else if isclass(py, gate_obj)? {
                        global_ideal_variable_width_gates.push(gate)
                    } else {
                        return Err(TranspilerError::new_err(
                            format!(
                                "The specified basis gate: {gate} has {gate_obj_num_qubits} \
                                qubits. This constructor method only supports fixed width operations \
                                with <= 2 qubits (because connectivity is defined on a CouplingMap)."
                            )
                        ));
                    }
                } else {
                    return Err(PyKeyError::new_err(format!(
                        "The specified basis gate: {gate} is not present in the standard gate names or a \
                        provided custom_name_mapping"
                    )));
                }
            }
            for gate in one_qubit_gates {
                let mut gate_properties: IndexMap<
                    Option<QargsTuple>,
                    Option<Py<InstructionProperties>>,
                > = IndexMap::new();
                for qubit in 0..num_qubits.unwrap_or_default() {
                    let mut error: Option<f64> = None;
                    let mut duration: Option<f64> = None;
                    let mut calibration: Option<Bound<PyAny>> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, qubit))
                            {
                                Ok(duration) => Some(duration.extract::<f64>()?),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties.call_method1("gate_error", (&gate, qubit))
                        {
                            Ok(error) => Some(error.extract::<f64>()?),
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
                        duration = match instruction_durations.call_method_bound(
                            py,
                            "get",
                            (&gate, qubit),
                            Some(&kwargs),
                        ) {
                            Ok(duration) => Some(duration.extract::<f64>(py)?),
                            Err(_) => None,
                        }
                    }
                    if error.is_none() && duration.is_none() && calibration.is_none() {
                        gate_properties
                            .insert(Some(smallvec![PhysicalQubit::new(qubit as u32)]), None);
                    } else {
                        gate_properties.insert(
                            Some(smallvec![PhysicalQubit::new(qubit as u32)]),
                            Some(Py::new(
                                py,
                                InstructionProperties::new(py, duration, error, calibration),
                            )?),
                        );
                    }
                }
                target.add_instruction(
                    py,
                    &name_mapping[&gate],
                    Some(gate_properties),
                    Some(gate),
                )?;
            }
            let edges = coupling_map
                .call_method0(py, "get_edges")?
                .extract::<Vec<[u32; 2]>>(py)?;
            for gate in two_qubit_gates {
                let mut gate_properties: IndexMap<
                    Option<QargsTuple>,
                    Option<Py<InstructionProperties>>,
                > = IndexMap::new();
                for edge in edges.as_slice().iter().cloned() {
                    let mut error: Option<f64> = None;
                    let mut duration: Option<f64> = None;
                    let mut calibration: Option<Bound<PyAny>> = None;
                    if let Some(backend_properties) = backend_properties {
                        if duration.is_none() {
                            duration = match backend_properties
                                .call_method1("gate_length", (&gate, edge))
                            {
                                Ok(duration) => Some(duration.extract::<f64>()?),
                                Err(_) => None,
                            }
                        }
                        error = match backend_properties.call_method1("gate_error", (&gate, edge)) {
                            Ok(error) => Some(error.extract::<f64>()?),
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
                        duration = match instruction_durations.call_method_bound(
                            py,
                            "get",
                            (&gate, edge),
                            Some(&kwargs),
                        ) {
                            Ok(duration) => Some(duration.extract::<f64>(py)?),
                            Err(_) => None,
                        }
                    }
                    if error.is_none() && duration.is_none() && calibration.is_none() {
                        gate_properties.insert(
                            Some(edge.into_iter().map(PhysicalQubit::new).collect()),
                            None,
                        );
                    } else {
                        gate_properties.insert(
                            Some(edge.into_iter().map(PhysicalQubit::new).collect()),
                            Some(Py::new(
                                py,
                                InstructionProperties::new(py, duration, error, calibration),
                            )?),
                        );
                    }
                }
                target.add_instruction(
                    py,
                    &name_mapping[&gate],
                    Some(gate_properties),
                    Some(gate),
                )?;
            }
            for gate in global_ideal_variable_width_gates {
                target.add_instruction(py, &name_mapping[&gate], None, Some(gate))?;
            }
        } else {
            for gate in basis_gates {
                if !name_mapping.contains_key(&gate) {
                    return Err(PyKeyError::new_err(format!(
                        "The specified basis gate: {gate} is not present in the standard gate \
                        names or a provided custom_name_mapping"
                    )));
                }
                target.add_instruction(py, &name_mapping[&gate], None, Some(gate))?;
            }
        }
        Ok(target)
    }

    // Magic methods:

    fn __iter__(&self) -> PyResult<Vec<String>> {
        Ok(self.gate_map.keys().cloned().collect())
    }

    fn __getitem__(&self, key: String) -> PyResult<QargPropMap> {
        if let Some(qarg_instprop) = self.gate_map.get(&key) {
            Ok(qarg_instprop.to_owned())
        } else {
            Err(PyKeyError::new_err(format!("{key} not in gate_map")))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: String,
        default: Option<Bound<PyAny>>,
    ) -> PyResult<PyObject> {
        match self.__getitem__(key) {
            Ok(value) => Ok(value.into_py(py)),
            Err(_) => Ok(match default {
                Some(value) => value.into(),
                None => py.None(),
            }),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __contains__(&self, item: String) -> PyResult<bool> {
        Ok(self.gate_map.contains_key(&item))
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
        result_list.append(self.gate_map.clone().into_py(py))?;
        result_list.append(self.gate_name_map.clone())?;
        result_list.append(self.global_operations.clone())?;
        result_list.append(self.qarg_gate_map.clone().into_py(py))?;
        result_list.append(self.coupling_graph.clone())?;
        result_list.append(self.instruction_durations.clone())?;
        result_list.append(self.instruction_schedule_map.clone())?;
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
        self.qubit_properties = state.get_item(7)?.extract::<Vec<PyObject>>()?;
        self.concurrent_measurements = state.get_item(8)?.extract::<Vec<Vec<usize>>>()?;
        self.gate_map = state.get_item(9)?.extract::<GateMapType>()?;
        self.gate_name_map = state
            .get_item(10)?
            .extract::<IndexMap<String, PyObject>>()?;
        self.global_operations = state
            .get_item(11)?
            .extract::<IndexMap<usize, HashSet<String>>>()?;
        self.qarg_gate_map = state
            .get_item(12)?
            .extract::<IndexMap<Option<Qargs>, Option<HashSet<String>>>>()?;
        self.coupling_graph = state.get_item(13)?.extract::<Option<PyObject>>()?;
        self.instruction_durations = state.get_item(14)?.extract::<Option<PyObject>>()?;
        self.instruction_schedule_map = state.get_item(15)?.extract::<Option<PyObject>>()?;
        self.non_global_basis = state.get_item(16)?.extract::<Option<Vec<String>>>()?;
        self.non_global_strict_basis = state.get_item(17)?.extract::<Option<Vec<String>>>()?;
        Ok(())
    }

    fn keys(&self) -> Vec<String> {
        self.gate_map.keys().cloned().collect()
    }

    fn values(&self) -> Vec<QargPropMap> {
        self.gate_map.values().cloned().collect_vec()
    }

    fn items(&self) -> Vec<(String, QargPropMap)> {
        self.gate_map.clone().into_iter().collect_vec()
    }
}

#[pymodule]
pub fn target(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}
