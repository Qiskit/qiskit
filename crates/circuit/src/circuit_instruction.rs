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

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PyTuple, PyType};
use pyo3::{intern, IntoPy, PyObject, PyResult};
use smallvec::SmallVec;

use crate::imports::{
    get_std_gate_class, populate_std_gate_map, GATE, INSTRUCTION, OPERATION,
    SINGLETON_CONTROLLED_GATE, SINGLETON_GATE,
};
use crate::operations::{OperationType, Param, PyGate, PyInstruction, PyOperation, StandardGate};

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
    pub params: Option<SmallVec<[Param; 3]>>,
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

#[pymethods]
impl CircuitInstruction {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        py: Python<'_>,
        operation: OperationInput,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        params: Option<SmallVec<[Param; 3]>>,
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
        let operation = match operation {
            Some(operation) => operation,
            None => match &self.operation {
                OperationType::Standard(op) => OperationInput::Standard(*op),
                OperationType::Gate(gate) => OperationInput::Gate(gate.clone()),
                OperationType::Instruction(inst) => OperationInput::Instruction(inst.clone()),
                OperationType::Operation(op) => OperationInput::Operation(op.clone()),
            },
        };

        let params = match params {
            Some(params) => Some(params),
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

        CircuitInstruction::new(
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
            [op, self.qubits.to_object(py), self.clbits.to_object(py)],
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
            [op, self.qubits.to_object(py), self.clbits.to_object(py)],
        ))
    }

    #[cfg(not(feature = "cache_pygates"))]
    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        Ok(self._legacy_format(py)?.as_any().get_item(key)?.into_py(py))
    }

    #[cfg(feature = "cache_pygates")]
    pub fn __getitem__(&mut self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        Ok(self._legacy_format(py)?.as_any().get_item(key)?.into_py(py))
    }

    #[cfg(not(feature = "cache_pygates"))]
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self._legacy_format(py)?.as_any().iter()?.into_py(py))
    }

    #[cfg(feature = "cache_pygates")]
    pub fn __iter__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self._legacy_format(py)?.as_any().iter()?.into_py(py))
    }

    pub fn __len__(&self) -> usize {
        3
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
                                } else if let Some(self_params) = &self_.params {
                                    if v.params.is_none() {
                                        return Ok(Some(false));
                                    }
                                    let other_params = v.params.as_ref().unwrap();
                                    let mut out = true;
                                    for (param_a, param_b) in self_params.iter().zip(other_params) {
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
                                } else {
                                    v.params.is_none()
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
pub(crate) fn operation_type_to_py(
    py: Python,
    circuit_inst: &CircuitInstruction,
) -> PyResult<PyObject> {
    let label;
    let duration;
    let unit;
    let condition;
    match &circuit_inst.extra_attrs {
        None => {
            label = None;
            duration = None;
            unit = None;
            condition = None;
        }
        Some(extra_attrs) => {
            label = extra_attrs.label.clone();
            duration = extra_attrs.duration.clone();
            unit = extra_attrs.unit.clone();
            condition = extra_attrs.condition.clone();
        }
    }
    operation_type_and_data_to_py(
        py,
        &circuit_inst.operation,
        &circuit_inst.params,
        &label,
        &duration,
        &unit,
        &condition,
    )
}

/// Take an OperationType and the other mutable state fields from a
/// rust instruction representation and return a PyObject representing
/// a Python side full-fat Qiskit operation as a PyObject. This is typically
/// used by accessor functions that need to return an operation to Qiskit, such
/// as accesing `CircuitInstruction.operation`.
pub(crate) fn operation_type_and_data_to_py(
    py: Python,
    operation: &OperationType,
    params: &Option<SmallVec<[Param; 3]>>,
    label: &Option<String>,
    duration: &Option<PyObject>,
    unit: &Option<String>,
    condition: &Option<PyObject>,
) -> PyResult<PyObject> {
    match &operation {
        OperationType::Standard(op) => {
            let gate_class: &PyObject = &get_std_gate_class(py, *op)?;

            let args = if let Some(params) = &params {
                if params.is_empty() {
                    PyTuple::empty_bound(py)
                } else {
                    PyTuple::new_bound(py, params)
                }
            } else {
                PyTuple::new_bound(py, params)
            };
            let mut kwargs_list = vec![
                ("label", label.to_object(py)),
                ("unit", unit.to_object(py)),
                ("duration", duration.to_object(py)),
            ];
            if let Some(params) = params {
                if !params.is_empty() {
                    kwargs_list.push(("_skip_validation", true.to_object(py)));
                }
            }

            let kwargs = kwargs_list.into_py_dict_bound(py);
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
#[derive(Debug)]
pub(crate) struct OperationTypeConstruct {
    pub operation: OperationType,
    pub params: Option<SmallVec<[Param; 3]>>,
    pub label: Option<String>,
    pub duration: Option<PyObject>,
    pub unit: Option<String>,
    pub condition: Option<PyObject>,
}

/// Convert an inbound Python object for a Qiskit operation and build a rust
/// representation of that operation. This will map it to appropriate variant
/// of operation type based on class
pub(crate) fn convert_py_to_operation_type(
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
    let mut standard: Option<StandardGate> = match op_type.getattr(attr).ok() {
        Some(stdgate) => match stdgate.extract().ok() {
            Some(gate) => gate,
            None => None,
        },
        None => None,
    };
    // If the input instruction is a standard gate and a singleton instance
    // we should check for mutable state. A mutable instance should be treated
    // as a custom gate not a standard gate because it has custom properties.
    //
    // In the futuer we can revisit this when we've dropped `duration`, `unit`,
    // and `condition` from the api as we should own the label in the
    // `CircuitInstruction`. The other piece here is for controlled gates there
    // is the control state, so for `SingletonControlledGates` we'll still need
    // this check.
    if standard.is_some() {
        let mutable: bool = py_op.getattr(py, intern!(py, "mutable"))?.extract(py)?;
        if mutable {
            let singleton_class = SINGLETON_GATE
                .get_or_init(py, || {
                    let singleton_mod = py.import_bound("qiskit.circuit.singleton").unwrap();
                    singleton_mod.getattr("SingletonGate").unwrap().unbind()
                })
                .bind(py);
            let singleton_control = SINGLETON_CONTROLLED_GATE
                .get_or_init(py, || {
                    let singleton_mod = py.import_bound("qiskit.circuit.singleton").unwrap();
                    singleton_mod
                        .getattr("SingletonControlledGate")
                        .unwrap()
                        .unbind()
                })
                .bind(py);

            if py_op_bound.is_instance(singleton_class)?
                || py_op_bound.is_instance(singleton_control)?
            {
                standard = None;
            }
        }
    }
    if let Some(op) = standard {
        let base_class = op_type.to_object(py);
        populate_std_gate_map(py, op, base_class);
        return Ok(OperationTypeConstruct {
            operation: OperationType::Standard(op),
            params: py_op
                .getattr(py, intern!(py, "params"))
                .ok()
                .unwrap()
                .extract(py)?,
            label: py_op
                .getattr(py, intern!(py, "label"))
                .ok()
                .unwrap()
                .extract(py)?,
            duration: py_op
                .getattr(py, intern!(py, "duration"))
                .ok()
                .unwrap()
                .extract(py)?,
            unit: py_op
                .getattr(py, intern!(py, "unit"))
                .ok()
                .unwrap()
                .extract(py)?,
            condition: py_op
                .getattr(py, intern!(py, "condition"))
                .ok()
                .unwrap()
                .extract(py)?,
        });
    }
    let gate_class = GATE
        .get_or_init(py, || {
            py.import_bound("qiskit.circuit.gate")
                .unwrap()
                .getattr("Gate")
                .unwrap()
                .unbind()
        })
        .bind(py);

    if op_type.is_subclass(gate_class)? {
        let params = py_op
            .getattr(py, intern!(py, "params"))
            .ok()
            .unwrap()
            .extract(py)?;
        let label = py_op
            .getattr(py, intern!(py, "label"))
            .ok()
            .unwrap()
            .extract(py)?;
        let duration = py_op
            .getattr(py, intern!(py, "duration"))
            .ok()
            .unwrap()
            .extract(py)?;
        let unit = py_op
            .getattr(py, intern!(py, "unit"))
            .ok()
            .unwrap()
            .extract(py)?;
        let condition = py_op
            .getattr(py, intern!(py, "condition"))
            .ok()
            .unwrap()
            .extract(py)?;

        let out_op = PyGate {
            qubits: py_op
                .getattr(py, intern!(py, "num_qubits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            clbits: py_op
                .getattr(py, intern!(py, "num_clbits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            params: py_op
                .getattr(py, intern!(py, "params"))
                .ok()
                .unwrap()
                .downcast_bound::<PyList>(py)?
                .len() as u32,
            op_name: py_op
                .getattr(py, intern!(py, "name"))
                .ok()
                .unwrap()
                .extract(py)?,
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
    let instruction_class = INSTRUCTION
        .get_or_init(py, || {
            py.import_bound("qiskit.circuit.instruction")
                .unwrap()
                .getattr("Instruction")
                .unwrap()
                .unbind()
        })
        .bind(py);
    if op_type.is_subclass(instruction_class)? {
        let params = py_op
            .getattr(py, intern!(py, "params"))
            .ok()
            .unwrap()
            .extract(py)?;
        let label = py_op
            .getattr(py, intern!(py, "label"))
            .ok()
            .unwrap()
            .extract(py)?;
        let duration = py_op
            .getattr(py, intern!(py, "duration"))
            .ok()
            .unwrap()
            .extract(py)?;
        let unit = py_op
            .getattr(py, intern!(py, "unit"))
            .ok()
            .unwrap()
            .extract(py)?;
        let condition = py_op
            .getattr(py, intern!(py, "condition"))
            .ok()
            .unwrap()
            .extract(py)?;

        let out_op = PyInstruction {
            qubits: py_op
                .getattr(py, intern!(py, "num_qubits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            clbits: py_op
                .getattr(py, intern!(py, "num_clbits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            params: py_op
                .getattr(py, intern!(py, "params"))
                .ok()
                .unwrap()
                .downcast_bound::<PyList>(py)?
                .len() as u32,
            op_name: py_op
                .getattr(py, intern!(py, "name"))
                .ok()
                .unwrap()
                .extract(py)?,
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

    let operation_class = OPERATION
        .get_or_init(py, || {
            py.import_bound("qiskit.circuit.operation")
                .unwrap()
                .getattr("Operation")
                .unwrap()
                .unbind()
        })
        .bind(py);
    if op_type.is_subclass(operation_class)? {
        let params = match py_op.getattr(py, intern!(py, "params")).ok() {
            Some(value) => value.extract(py)?,
            None => None,
        };
        let label = None;
        let duration = None;
        let unit = None;
        let condition = None;
        let out_op = PyOperation {
            qubits: py_op
                .getattr(py, intern!(py, "num_qubits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            clbits: py_op
                .getattr(py, intern!(py, "num_clbits"))
                .ok()
                .map(|x| x.extract(py).unwrap())
                .unwrap_or(0),
            params: match py_op.getattr(py, intern!(py, "params")).ok() {
                Some(value) => value.downcast_bound::<PyList>(py)?.len() as u32,
                None => 0,
            },
            op_name: py_op
                .getattr(py, intern!(py, "name"))
                .ok()
                .unwrap()
                .extract(py)?,
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
