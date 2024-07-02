// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::cell::RefCell;

use numpy::IntoPyArray;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyDeprecationWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PyTuple, PyType};
use pyo3::{intern, IntoPy, PyObject, PyResult};
use smallvec::{smallvec, SmallVec};

use crate::imports::{
    get_std_gate_class, populate_std_gate_map, CONTROLLED_GATE, GATE, INSTRUCTION, OPERATION,
    SINGLETON_CONTROLLED_GATE, SINGLETON_GATE, WARNINGS_WARN,
};
use crate::interner::Index;
use crate::operations::{
    Operation, OperationType, Param, PyGate, PyInstruction, PyOperation, StandardGate,
};

/// These are extra mutable attributes for a circuit instruction's state. In general we don't
/// typically deal with this in rust space and the majority of the time they're not used in Python
/// space either. To save memory these are put in a separate struct and are stored inside a
/// `Box` on `CircuitInstruction` and `PackedInstruction`.
#[derive(Debug, Clone)]
pub struct ExtraInstructionAttributes {
    pub label: Option<String>,
    pub duration: Option<PyObject>,
    pub unit: Option<String>,
    pub condition: Option<PyObject>,
}

/// Private type used to store instructions with interned arg lists.
#[derive(Clone, Debug)]
pub(crate) struct PackedInstruction {
    /// The Python-side operation instance.
    pub op: OperationType,
    /// The index under which the interner has stored `qubits`.
    pub qubits_id: Index,
    /// The index under which the interner has stored `clbits`.
    pub clbits_id: Index,
    pub params: SmallVec<[Param; 3]>,
    pub extra_attrs: Option<Box<ExtraInstructionAttributes>>,

    #[cfg(feature = "cache_pygates")]
    /// This is hidden in a `RefCell` because, while that has additional memory-usage implications
    /// while we're still building with the feature enabled, we intend to remove the feature in the
    /// future, and hiding the cache within a `RefCell` lets us keep the cache transparently in our
    /// interfaces, without needing various functions to unnecessarily take `&mut` references.
    pub py_op: RefCell<Option<PyObject>>,
}

impl PackedInstruction {
    /// Build a reference to the Python-space operation object (the `Gate`, etc) packed into this
    /// instruction.  This may construct the reference if the `PackedInstruction` is a standard
    /// gate with no already stored operation.
    ///
    /// A standard-gate operation object returned by this function is disconnected from the
    /// containing circuit; updates to its label, duration, unit and condition will not be
    /// propagated back.
    pub fn unpack_py_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        #[cfg(feature = "cache_pygates")]
        {
            if let Some(cached_op) = self.py_op.borrow().as_ref() {
                return Ok(cached_op.clone_ref(py));
            }
        }
        let (label, duration, unit, condition) = match self.extra_attrs.as_deref() {
            Some(ExtraInstructionAttributes {
                label,
                duration,
                unit,
                condition,
            }) => (
                label.as_deref(),
                duration.as_ref(),
                unit.as_deref(),
                condition.as_ref(),
            ),
            None => (None, None, None, None),
        };
        let out = operation_type_and_data_to_py(
            py,
            &self.op,
            &self.params,
            label,
            duration,
            unit,
            condition,
        )?;
        #[cfg(feature = "cache_pygates")]
        {
            if let Ok(mut cell) = self.py_op.try_borrow_mut() {
                cell.get_or_insert_with(|| out.clone_ref(py));
            }
        }
        Ok(out)
    }
}

/// A single instruction in a :class:`.QuantumCircuit`, comprised of the :attr:`operation` and
/// various operands.
///
/// .. note::
///
///     There is some possible confusion in the names of this class, :class:`~.circuit.Instruction`,
///     and :class:`~.circuit.Operation`, and this class's attribute :attr:`operation`.  Our
///     preferred terminology is by analogy to assembly languages, where an "instruction" is made up
///     of an "operation" and its "operands".
///
///     Historically, :class:`~.circuit.Instruction` came first, and originally contained the qubits
///     it operated on and any parameters, so it was a true "instruction".  Over time,
///     :class:`.QuantumCircuit` became responsible for tracking qubits and clbits, and the class
///     became better described as an "operation".  Changing the name of such a core object would be
///     a very unpleasant API break for users, and so we have stuck with it.
///
///     This class was created to provide a formal "instruction" context object in
///     :class:`.QuantumCircuit.data`, which had long been made of ad-hoc tuples.  With this, and
///     the advent of the :class:`~.circuit.Operation` interface for adding more complex objects to
///     circuits, we took the opportunity to correct the historical naming.  For the time being,
///     this leads to an awkward case where :attr:`.CircuitInstruction.operation` is often an
///     :class:`~.circuit.Instruction` instance (:class:`~.circuit.Instruction` implements the
///     :class:`.Operation` interface), but as the :class:`.Operation` interface gains more use,
///     this confusion will hopefully abate.
///
/// .. warning::
///
///     This is a lightweight internal class and there is minimal error checking; you must respect
///     the type hints when using it.  It is the user's responsibility to ensure that direct
///     mutations of the object do not invalidate the types, nor the restrictions placed on it by
///     its context.  Typically this will mean, for example, that :attr:`qubits` must be a sequence
///     of distinct items, with no duplicates.
#[pyclass(freelist = 20, sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct CircuitInstruction {
    pub operation: OperationType,
    /// A sequence of the qubits that the operation is applied to.
    #[pyo3(get)]
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    #[pyo3(get)]
    pub clbits: Py<PyTuple>,
    pub params: SmallVec<[Param; 3]>,
    pub extra_attrs: Option<Box<ExtraInstructionAttributes>>,
    #[cfg(feature = "cache_pygates")]
    pub py_op: Option<PyObject>,
}

/// This enum is for backwards compatibility if a user was doing something from
/// Python like CircuitInstruction(SXGate(), [qr[0]], []) by passing a python
/// gate object directly to a CircuitInstruction. In this case we need to
/// create a rust side object from the pyobject in CircuitInstruction.new()
/// With the `Object` variant which will convert the python object to a rust
/// `OperationType`
#[derive(FromPyObject, Debug)]
pub enum OperationInput {
    Standard(StandardGate),
    Gate(PyGate),
    Instruction(PyInstruction),
    Operation(PyOperation),
    Object(PyObject),
}

impl CircuitInstruction {
    pub fn new<T1, T2, U1, U2>(
        py: Python,
        operation: OperationType,
        qubits: impl IntoIterator<Item = T1, IntoIter = U1>,
        clbits: impl IntoIterator<Item = T2, IntoIter = U2>,
        params: SmallVec<[Param; 3]>,
        extra_attrs: Option<Box<ExtraInstructionAttributes>>,
    ) -> Self
    where
        T1: ToPyObject,
        T2: ToPyObject,
        U1: ExactSizeIterator<Item = T1>,
        U2: ExactSizeIterator<Item = T2>,
    {
        CircuitInstruction {
            operation,
            qubits: PyTuple::new_bound(py, qubits).unbind(),
            clbits: PyTuple::new_bound(py, clbits).unbind(),
            params,
            extra_attrs,
            #[cfg(feature = "cache_pygates")]
            py_op: None,
        }
    }
}

impl From<OperationType> for OperationInput {
    fn from(value: OperationType) -> Self {
        match value {
            OperationType::Standard(op) => Self::Standard(op),
            OperationType::Gate(gate) => Self::Gate(gate),
            OperationType::Instruction(inst) => Self::Instruction(inst),
            OperationType::Operation(op) => Self::Operation(op),
        }
    }
}

#[pymethods]
impl CircuitInstruction {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (operation, qubits=None, clbits=None, params=smallvec![], label=None, duration=None, unit=None, condition=None))]
    pub fn py_new(
        py: Python<'_>,
        operation: OperationInput,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        params: SmallVec<[Param; 3]>,
        label: Option<String>,
        duration: Option<PyObject>,
        unit: Option<String>,
        condition: Option<PyObject>,
    ) -> PyResult<Self> {
        fn as_tuple(py: Python<'_>, seq: Option<&Bound<PyAny>>) -> PyResult<Py<PyTuple>> {
            match seq {
                None => Ok(PyTuple::empty_bound(py).unbind()),
                Some(seq) => {
                    if seq.is_instance_of::<PyTuple>() {
                        Ok(seq.downcast_exact::<PyTuple>()?.into_py(py))
                    } else if seq.is_instance_of::<PyList>() {
                        let seq = seq.downcast_exact::<PyList>()?;
                        Ok(seq.to_tuple().unbind())
                    } else {
                        // New tuple from iterable.
                        Ok(PyTuple::new_bound(
                            py,
                            seq.iter()?
                                .map(|o| Ok(o?.unbind()))
                                .collect::<PyResult<Vec<PyObject>>>()?,
                        )
                        .unbind())
                    }
                }
            }
        }

        let extra_attrs =
            if label.is_some() || duration.is_some() || unit.is_some() || condition.is_some() {
                Some(Box::new(ExtraInstructionAttributes {
                    label,
                    duration,
                    unit,
                    condition,
                }))
            } else {
                None
            };

        match operation {
            OperationInput::Standard(operation) => {
                let operation = OperationType::Standard(operation);
                Ok(CircuitInstruction {
                    operation,
                    qubits: as_tuple(py, qubits)?,
                    clbits: as_tuple(py, clbits)?,
                    params,
                    extra_attrs,
                    #[cfg(feature = "cache_pygates")]
                    py_op: None,
                })
            }
            OperationInput::Gate(operation) => {
                let operation = OperationType::Gate(operation);
                Ok(CircuitInstruction {
                    operation,
                    qubits: as_tuple(py, qubits)?,
                    clbits: as_tuple(py, clbits)?,
                    params,
                    extra_attrs,
                    #[cfg(feature = "cache_pygates")]
                    py_op: None,
                })
            }
            OperationInput::Instruction(operation) => {
                let operation = OperationType::Instruction(operation);
                Ok(CircuitInstruction {
                    operation,
                    qubits: as_tuple(py, qubits)?,
                    clbits: as_tuple(py, clbits)?,
                    params,
                    extra_attrs,
                    #[cfg(feature = "cache_pygates")]
                    py_op: None,
                })
            }
            OperationInput::Operation(operation) => {
                let operation = OperationType::Operation(operation);
                Ok(CircuitInstruction {
                    operation,
                    qubits: as_tuple(py, qubits)?,
                    clbits: as_tuple(py, clbits)?,
                    params,
                    extra_attrs,
                    #[cfg(feature = "cache_pygates")]
                    py_op: None,
                })
            }
            OperationInput::Object(old_op) => {
                let op = convert_py_to_operation_type(py, old_op.clone_ref(py))?;
                let extra_attrs = if op.label.is_some()
                    || op.duration.is_some()
                    || op.unit.is_some()
                    || op.condition.is_some()
                {
                    Some(Box::new(ExtraInstructionAttributes {
                        label: op.label,
                        duration: op.duration,
                        unit: op.unit,
                        condition: op.condition,
                    }))
                } else {
                    None
                };

                match op.operation {
                    OperationType::Standard(operation) => {
                        let operation = OperationType::Standard(operation);
                        Ok(CircuitInstruction {
                            operation,
                            qubits: as_tuple(py, qubits)?,
                            clbits: as_tuple(py, clbits)?,
                            params: op.params,
                            extra_attrs,
                            #[cfg(feature = "cache_pygates")]
                            py_op: Some(old_op.clone_ref(py)),
                        })
                    }
                    OperationType::Gate(operation) => {
                        let operation = OperationType::Gate(operation);
                        Ok(CircuitInstruction {
                            operation,
                            qubits: as_tuple(py, qubits)?,
                            clbits: as_tuple(py, clbits)?,
                            params: op.params,
                            extra_attrs,
                            #[cfg(feature = "cache_pygates")]
                            py_op: Some(old_op.clone_ref(py)),
                        })
                    }
                    OperationType::Instruction(operation) => {
                        let operation = OperationType::Instruction(operation);
                        Ok(CircuitInstruction {
                            operation,
                            qubits: as_tuple(py, qubits)?,
                            clbits: as_tuple(py, clbits)?,
                            params: op.params,
                            extra_attrs,
                            #[cfg(feature = "cache_pygates")]
                            py_op: Some(old_op.clone_ref(py)),
                        })
                    }
                    OperationType::Operation(operation) => {
                        let operation = OperationType::Operation(operation);
                        Ok(CircuitInstruction {
                            operation,
                            qubits: as_tuple(py, qubits)?,
                            clbits: as_tuple(py, clbits)?,
                            params: op.params,
                            extra_attrs,
                            #[cfg(feature = "cache_pygates")]
                            py_op: Some(old_op.clone_ref(py)),
                        })
                    }
                }
            }
        }
    }

    /// Returns a shallow copy.
    ///
    /// Returns:
    ///     CircuitInstruction: The shallow copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// The logical operation that this instruction represents an execution of.
    #[cfg(not(feature = "cache_pygates"))]
    #[getter]
    pub fn operation(&self, py: Python) -> PyResult<PyObject> {
        operation_type_to_py(py, self)
    }

    #[cfg(feature = "cache_pygates")]
    #[getter]
    pub fn operation(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(match &self.py_op {
            Some(op) => op.clone_ref(py),
            None => {
                let op = operation_type_to_py(py, self)?;
                self.py_op = Some(op.clone_ref(py));
                op
            }
        })
    }

    #[getter]
    fn _raw_op(&self, py: Python) -> PyObject {
        self.operation.clone().into_py(py)
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self, py: Python) -> PyObject {
        self.operation.name().to_object(py)
    }

    #[getter]
    fn get_params(&self, py: Python) -> PyObject {
        self.params.to_object(py)
    }

    #[getter]
    fn matrix(&self, py: Python) -> Option<PyObject> {
        let matrix = self.operation.matrix(&self.params);
        matrix.map(|mat| mat.into_pyarray_bound(py).into())
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.label.as_deref())
    }

    #[getter]
    fn condition(&self, py: Python) -> Option<PyObject> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.condition.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn duration(&self, py: Python) -> Option<PyObject> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.duration.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn unit(&self) -> Option<&str> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.unit.as_deref())
    }

    /// Creates a shallow copy with the given fields replaced.
    ///
    /// Returns:
    ///     CircuitInstruction: A new instance with the given fields replaced.
    #[allow(clippy::too_many_arguments)]
    pub fn replace(
        &self,
        py: Python<'_>,
        operation: Option<OperationInput>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<String>,
        duration: Option<PyObject>,
        unit: Option<String>,
        condition: Option<PyObject>,
    ) -> PyResult<Self> {
        let operation = operation.unwrap_or_else(|| self.operation.clone().into());

        let params = match params {
            Some(params) => params,
            None => self.params.clone(),
        };

        let label = match label {
            Some(label) => Some(label),
            None => match &self.extra_attrs {
                Some(extra_attrs) => extra_attrs.label.clone(),
                None => None,
            },
        };
        let duration = match duration {
            Some(duration) => Some(duration),
            None => match &self.extra_attrs {
                Some(extra_attrs) => extra_attrs.duration.clone(),
                None => None,
            },
        };

        let unit: Option<String> = match unit {
            Some(unit) => Some(unit),
            None => match &self.extra_attrs {
                Some(extra_attrs) => extra_attrs.unit.clone(),
                None => None,
            },
        };

        let condition: Option<PyObject> = match condition {
            Some(condition) => Some(condition),
            None => match &self.extra_attrs {
                Some(extra_attrs) => extra_attrs.condition.clone(),
                None => None,
            },
        };

        CircuitInstruction::py_new(
            py,
            operation,
            Some(qubits.unwrap_or_else(|| self.qubits.bind(py))),
            Some(clbits.unwrap_or_else(|| self.clbits.bind(py))),
            params,
            label,
            duration,
            unit,
            condition,
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            operation_type_to_py(py, self)?,
            self.qubits.bind(py),
            self.clbits.bind(py),
        )
            .into_py(py))
    }

    fn __setstate__(&mut self, py: Python<'_>, state: &Bound<PyTuple>) -> PyResult<()> {
        let op = convert_py_to_operation_type(py, state.get_item(0)?.into())?;
        self.operation = op.operation;
        self.params = op.params;
        self.qubits = state.get_item(1)?.extract()?;
        self.clbits = state.get_item(2)?.extract()?;
        if op.label.is_some()
            || op.duration.is_some()
            || op.unit.is_some()
            || op.condition.is_some()
        {
            self.extra_attrs = Some(Box::new(ExtraInstructionAttributes {
                label: op.label,
                duration: op.duration,
                unit: op.unit,
                condition: op.condition,
            }));
        }
        Ok(())
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            operation_type_to_py(py, self)?,
            self.qubits.bind(py),
            self.clbits.bind(py),
        )
            .into_py(py))
    }

    pub fn __repr__(self_: &Bound<Self>, py: Python<'_>) -> PyResult<String> {
        let type_name = self_.get_type().qualname()?;
        let r = self_.try_borrow()?;
        Ok(format!(
            "{}(\
            operation={}\
            , qubits={}\
            , clbits={}\
            )",
            type_name,
            operation_type_to_py(py, &r)?,
            r.qubits.bind(py).repr()?,
            r.clbits.bind(py).repr()?
        ))
    }

    // Legacy tuple-like interface support.
    //
    // For a best attempt at API compatibility during the transition to using this new class, we need
    // the interface to behave exactly like the old 3-tuple `(inst, qargs, cargs)` if it's treated
    // like that via unpacking or similar.  That means that the `parameters` field is completely
    // absent, and the qubits and clbits must be converted to lists.
    #[cfg(not(feature = "cache_pygates"))]
    pub fn _legacy_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let op = operation_type_to_py(py, self)?;

        Ok(PyTuple::new_bound(
            py,
            [
                op,
                self.qubits.bind(py).to_list().into(),
                self.clbits.bind(py).to_list().into(),
            ],
        ))
    }

    #[cfg(feature = "cache_pygates")]
    pub fn _legacy_format<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let op = match &self.py_op {
            Some(op) => op.clone_ref(py),
            None => {
                let op = operation_type_to_py(py, self)?;
                self.py_op = Some(op.clone_ref(py));
                op
            }
        };
        Ok(PyTuple::new_bound(
            py,
            [
                op,
                self.qubits.bind(py).to_list().into(),
                self.clbits.bind(py).to_list().into(),
            ],
        ))
    }

    #[cfg(not(feature = "cache_pygates"))]
    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().get_item(key)?.into_py(py))
    }

    #[cfg(feature = "cache_pygates")]
    pub fn __getitem__(&mut self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().get_item(key)?.into_py(py))
    }

    #[cfg(not(feature = "cache_pygates"))]
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().iter()?.into_py(py))
    }

    #[cfg(feature = "cache_pygates")]
    pub fn __iter__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().iter()?.into_py(py))
    }

    pub fn __len__(&self, py: Python) -> PyResult<usize> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(3)
    }

    pub fn __richcmp__(
        self_: &Bound<Self>,
        other: &Bound<PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        fn eq(
            py: Python<'_>,
            self_: &Bound<CircuitInstruction>,
            other: &Bound<PyAny>,
        ) -> PyResult<Option<bool>> {
            if self_.is(other) {
                return Ok(Some(true));
            }

            let self_ = self_.try_borrow()?;
            if other.is_instance_of::<CircuitInstruction>() {
                let other: PyResult<Bound<CircuitInstruction>> = other.extract();
                return other.map_or(Ok(Some(false)), |v| {
                    let v = v.try_borrow()?;
                    let op_eq = match &self_.operation {
                        OperationType::Standard(op) => {
                            if let OperationType::Standard(other) = &v.operation {
                                if op != other {
                                    false
                                } else {
                                    let other_params = &v.params;
                                    let mut out = true;
                                    for (param_a, param_b) in self_.params.iter().zip(other_params)
                                    {
                                        match param_a {
                                            Param::Float(val_a) => {
                                                if let Param::Float(val_b) = param_b {
                                                    if val_a != val_b {
                                                        out = false;
                                                        break;
                                                    }
                                                } else {
                                                    out = false;
                                                    break;
                                                }
                                            }
                                            Param::ParameterExpression(val_a) => {
                                                if let Param::ParameterExpression(val_b) = param_b {
                                                    if !val_a.bind(py).eq(val_b.bind(py))? {
                                                        out = false;
                                                        break;
                                                    }
                                                } else {
                                                    out = false;
                                                    break;
                                                }
                                            }
                                            Param::Obj(val_a) => {
                                                if let Param::Obj(val_b) = param_b {
                                                    if !val_a.bind(py).eq(val_b.bind(py))? {
                                                        out = false;
                                                        break;
                                                    }
                                                } else {
                                                    out = false;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    out
                                }
                            } else {
                                false
                            }
                        }
                        OperationType::Gate(op) => {
                            if let OperationType::Gate(other) = &v.operation {
                                op.gate.bind(py).eq(other.gate.bind(py))?
                            } else {
                                false
                            }
                        }
                        OperationType::Instruction(op) => {
                            if let OperationType::Instruction(other) = &v.operation {
                                op.instruction.bind(py).eq(other.instruction.bind(py))?
                            } else {
                                false
                            }
                        }
                        OperationType::Operation(op) => {
                            if let OperationType::Operation(other) = &v.operation {
                                op.operation.bind(py).eq(other.operation.bind(py))?
                            } else {
                                false
                            }
                        }
                    };

                    Ok(Some(
                        self_.clbits.bind(py).eq(v.clbits.bind(py))?
                            && self_.qubits.bind(py).eq(v.qubits.bind(py))?
                            && op_eq,
                    ))
                });
            }

            if other.is_instance_of::<PyTuple>() {
                #[cfg(feature = "cache_pygates")]
                let mut self_ = self_.clone();
                let legacy_format = self_._legacy_format(py)?;
                return Ok(Some(legacy_format.eq(other)?));
            }

            Ok(None)
        }

        match op {
            CompareOp::Eq => eq(py, self_, other).map(|r| {
                r.map(|b| b.into_py(py))
                    .unwrap_or_else(|| py.NotImplemented())
            }),
            CompareOp::Ne => eq(py, self_, other).map(|r| {
                r.map(|b| (!b).into_py(py))
                    .unwrap_or_else(|| py.NotImplemented())
            }),
            _ => Ok(py.NotImplemented()),
        }
    }
}

/// Take a reference to a `CircuitInstruction` and convert the operation
/// inside that to a python side object.
pub fn operation_type_to_py(py: Python, circuit_inst: &CircuitInstruction) -> PyResult<PyObject> {
    let (label, duration, unit, condition) = match &circuit_inst.extra_attrs {
        None => (None, None, None, None),
        Some(extra_attrs) => (
            extra_attrs.label.as_deref(),
            extra_attrs.duration.as_ref(),
            extra_attrs.unit.as_deref(),
            extra_attrs.condition.as_ref(),
        ),
    };
    operation_type_and_data_to_py(
        py,
        &circuit_inst.operation,
        &circuit_inst.params,
        label,
        duration,
        unit,
        condition,
    )
}

/// Take an OperationType and the other mutable state fields from a
/// rust instruction representation and return a PyObject representing
/// a Python side full-fat Qiskit operation as a PyObject. This is typically
/// used by accessor functions that need to return an operation to Qiskit, such
/// as accesing `CircuitInstruction.operation`.
pub fn operation_type_and_data_to_py(
    py: Python,
    operation: &OperationType,
    params: &[Param],
    label: Option<&str>,
    duration: Option<&PyObject>,
    unit: Option<&str>,
    condition: Option<&PyObject>,
) -> PyResult<PyObject> {
    match &operation {
        OperationType::Standard(op) => {
            let gate_class: &PyObject = &get_std_gate_class(py, *op)?;

            let args = if params.is_empty() {
                PyTuple::empty_bound(py)
            } else {
                PyTuple::new_bound(py, params)
            };
            let kwargs = [
                ("label", label.to_object(py)),
                ("unit", unit.to_object(py)),
                ("duration", duration.to_object(py)),
            ]
            .into_py_dict_bound(py);
            let mut out = gate_class.call_bound(py, args, Some(&kwargs))?;
            if condition.is_some() {
                out = out.call_method0(py, "to_mutable")?;
                out.setattr(py, "condition", condition.to_object(py))?;
            }
            Ok(out)
        }
        OperationType::Gate(gate) => Ok(gate.gate.clone_ref(py)),
        OperationType::Instruction(inst) => Ok(inst.instruction.clone_ref(py)),
        OperationType::Operation(op) => Ok(op.operation.clone_ref(py)),
    }
}

/// A container struct that contains the output from the Python object to
/// conversion to construct a CircuitInstruction object
#[derive(Debug, Clone)]
pub struct OperationTypeConstruct {
    pub operation: OperationType,
    pub params: SmallVec<[Param; 3]>,
    pub label: Option<String>,
    pub duration: Option<PyObject>,
    pub unit: Option<String>,
    pub condition: Option<PyObject>,
}

/// Convert an inbound Python object for a Qiskit operation and build a rust
/// representation of that operation. This will map it to appropriate variant
/// of operation type based on class
pub fn convert_py_to_operation_type(
    py: Python,
    py_op: PyObject,
) -> PyResult<OperationTypeConstruct> {
    let attr = intern!(py, "_standard_gate");
    let py_op_bound = py_op.clone_ref(py).into_bound(py);
    // Get PyType from either base_class if it exists, or if not use the
    // class/type info from the pyobject
    let binding = py_op_bound.getattr(intern!(py, "base_class")).ok();
    let op_obj = py_op_bound.get_type();
    let raw_op_type: Py<PyType> = match binding {
        Some(base_class) => base_class.downcast()?.clone().unbind(),
        None => op_obj.unbind(),
    };
    let op_type: Bound<PyType> = raw_op_type.into_bound(py);
    let mut standard: Option<StandardGate> = match op_type.getattr(attr) {
        Ok(stdgate) => stdgate.extract().ok().unwrap_or_default(),
        Err(_) => None,
    };
    // If the input instruction is a standard gate and a singleton instance,
    // we should check for mutable state. A mutable instance should be treated
    // as a custom gate not a standard gate because it has custom properties.
    // Controlled gates with non-default control states are also considered
    // custom gates even if a standard representation exists for the default
    // control state.

    // In the future we can revisit this when we've dropped `duration`, `unit`,
    // and `condition` from the api as we should own the label in the
    // `CircuitInstruction`. The other piece here is for controlled gates there
    // is the control state, so for `SingletonControlledGates` we'll still need
    // this check.
    if standard.is_some() {
        let mutable: bool = py_op.getattr(py, intern!(py, "mutable"))?.extract(py)?;
        // The default ctrl_states are 1, "1" and None. These are the only cases where controlled
        // gates can be standard.
        let is_default_ctrl_state: bool = match py_op.getattr(py, intern!(py, "ctrl_state")) {
            Ok(c_state) => match c_state.extract::<Option<i32>>(py) {
                Ok(c_state_int) => match c_state_int {
                    Some(c_int) => c_int == 1,
                    None => true,
                },
                Err(_) => false,
            },
            Err(_) => false,
        };

        if (mutable
            && (py_op_bound.is_instance(SINGLETON_GATE.get_bound(py))?
                || py_op_bound.is_instance(SINGLETON_CONTROLLED_GATE.get_bound(py))?))
            || (py_op_bound.is_instance(CONTROLLED_GATE.get_bound(py))? && !is_default_ctrl_state)
        {
            standard = None;
        }
    }
    if let Some(op) = standard {
        let base_class = op_type.to_object(py);
        populate_std_gate_map(py, op, base_class);
        return Ok(OperationTypeConstruct {
            operation: OperationType::Standard(op),
            params: py_op.getattr(py, intern!(py, "params"))?.extract(py)?,
            label: py_op.getattr(py, intern!(py, "label"))?.extract(py)?,
            duration: py_op.getattr(py, intern!(py, "duration"))?.extract(py)?,
            unit: py_op.getattr(py, intern!(py, "unit"))?.extract(py)?,
            condition: py_op.getattr(py, intern!(py, "condition"))?.extract(py)?,
        });
    }
    if op_type.is_subclass(GATE.get_bound(py))? {
        let params = py_op.getattr(py, intern!(py, "params"))?.extract(py)?;
        let label = py_op.getattr(py, intern!(py, "label"))?.extract(py)?;
        let duration = py_op.getattr(py, intern!(py, "duration"))?.extract(py)?;
        let unit = py_op.getattr(py, intern!(py, "unit"))?.extract(py)?;
        let condition = py_op.getattr(py, intern!(py, "condition"))?.extract(py)?;

        let out_op = PyGate {
            qubits: py_op.getattr(py, intern!(py, "num_qubits"))?.extract(py)?,
            clbits: py_op.getattr(py, intern!(py, "num_clbits"))?.extract(py)?,
            params: py_op
                .getattr(py, intern!(py, "params"))?
                .downcast_bound::<PyList>(py)?
                .len() as u32,
            op_name: py_op.getattr(py, intern!(py, "name"))?.extract(py)?,
            gate: py_op,
        };
        return Ok(OperationTypeConstruct {
            operation: OperationType::Gate(out_op),
            params,
            label,
            duration,
            unit,
            condition,
        });
    }
    if op_type.is_subclass(INSTRUCTION.get_bound(py))? {
        let params = py_op.getattr(py, intern!(py, "params"))?.extract(py)?;
        let label = py_op.getattr(py, intern!(py, "label"))?.extract(py)?;
        let duration = py_op.getattr(py, intern!(py, "duration"))?.extract(py)?;
        let unit = py_op.getattr(py, intern!(py, "unit"))?.extract(py)?;
        let condition = py_op.getattr(py, intern!(py, "condition"))?.extract(py)?;

        let out_op = PyInstruction {
            qubits: py_op.getattr(py, intern!(py, "num_qubits"))?.extract(py)?,
            clbits: py_op.getattr(py, intern!(py, "num_clbits"))?.extract(py)?,
            params: py_op
                .getattr(py, intern!(py, "params"))?
                .downcast_bound::<PyList>(py)?
                .len() as u32,
            op_name: py_op.getattr(py, intern!(py, "name"))?.extract(py)?,
            instruction: py_op,
        };
        return Ok(OperationTypeConstruct {
            operation: OperationType::Instruction(out_op),
            params,
            label,
            duration,
            unit,
            condition,
        });
    }

    if op_type.is_subclass(OPERATION.get_bound(py))? {
        let params = match py_op.getattr(py, intern!(py, "params")) {
            Ok(value) => value.extract(py)?,
            Err(_) => smallvec![],
        };
        let label = None;
        let duration = None;
        let unit = None;
        let condition = None;
        let out_op = PyOperation {
            qubits: py_op.getattr(py, intern!(py, "num_qubits"))?.extract(py)?,
            clbits: py_op.getattr(py, intern!(py, "num_clbits"))?.extract(py)?,
            params: match py_op.getattr(py, intern!(py, "params")) {
                Ok(value) => value.downcast_bound::<PyList>(py)?.len() as u32,
                Err(_) => 0,
            },
            op_name: py_op.getattr(py, intern!(py, "name"))?.extract(py)?,
            operation: py_op,
        };
        return Ok(OperationTypeConstruct {
            operation: OperationType::Operation(out_op),
            params,
            label,
            duration,
            unit,
            condition,
        });
    }
    Err(PyValueError::new_err(format!("Invalid input: {}", py_op)))
}

/// Issue a Python `DeprecationWarning` about using the legacy tuple-like interface to
/// `CircuitInstruction`.
///
/// Beware the `stacklevel` here doesn't work quite the same way as it does in Python as Rust-space
/// calls are completely transparent to Python.
#[inline]
fn warn_on_legacy_circuit_instruction_iteration(py: Python) -> PyResult<()> {
    WARNINGS_WARN
        .get_bound(py)
        .call1((
            intern!(
                py,
                concat!(
                    "Treating CircuitInstruction as an iterable is deprecated legacy behavior",
                    " since Qiskit 1.2, and will be removed in Qiskit 2.0.",
                    " Instead, use the `operation`, `qubits` and `clbits` named attributes."
                )
            ),
            py.get_type_bound::<PyDeprecationWarning>(),
            // Stack level.  Compared to Python-space calls to `warn`, this is unusually low
            // beacuse all our internal call structure is now Rust-space and invisible to Python.
            1,
        ))
        .map(|_| ())
}
