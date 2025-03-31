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
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyDeprecationWarning, PyTypeError};
use pyo3::prelude::*;

use pyo3::types::{PyBool, PyList, PyTuple, PyType};
use pyo3::IntoPyObjectExt;
use pyo3::{intern, PyObject, PyResult};

use nalgebra::{MatrixView2, MatrixView4};
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::imports::{
    CONTROLLED_GATE, CONTROL_FLOW_OP, GATE, INSTRUCTION, OPERATION, WARNINGS_WARN,
};
use crate::operations::{
    ArrayType, Operation, OperationRef, Param, PyGate, PyInstruction, PyOperation, StandardGate,
    StandardInstruction, StandardInstructionType, UnitaryGate,
};
use crate::packed_instruction::PackedOperation;

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
    pub operation: PackedOperation,
    /// A sequence of the qubits that the operation is applied to.
    #[pyo3(get)]
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    #[pyo3(get)]
    pub clbits: Py<PyTuple>,
    pub params: SmallVec<[Param; 3]>,
    pub label: Option<Box<String>>,
    #[cfg(feature = "cache_pygates")]
    pub py_op: OnceLock<Py<PyAny>>,
}

impl CircuitInstruction {
    /// Get the Python-space operation, ensuring that it is mutable from Python space (singleton
    /// gates might not necessarily satisfy this otherwise).
    ///
    /// This returns the cached instruction if valid, but does not replace the cache if it created a
    /// new mutable object; the expectation is that any mutations to the Python object need
    /// assigning back to the `CircuitInstruction` completely to ensure data coherence between Rust
    /// and Python spaces.  We can't protect entirely against that, but we can make it a bit harder
    /// for standard-gate getters to accidentally do the wrong thing.
    pub fn get_operation_mut<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = self.get_operation(py)?.into_bound(py);
        if out.getattr(intern!(py, "mutable"))?.is_truthy()? {
            Ok(out)
        } else {
            out.call_method0(intern!(py, "to_mutable"))
        }
    }
}

#[pymethods]
impl CircuitInstruction {
    #[new]
    #[pyo3(signature = (operation, qubits=None, clbits=None))]
    pub fn py_new(
        operation: &Bound<PyAny>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let py = operation.py();
        let op_parts = operation.extract::<OperationFromPython>()?;

        Ok(Self {
            operation: op_parts.operation,
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: as_tuple(py, clbits)?.unbind(),
            params: op_parts.params,
            label: op_parts.label,
            #[cfg(feature = "cache_pygates")]
            py_op: operation.clone().unbind().into(),
        })
    }

    #[pyo3(signature = (standard, qubits, params, label=None))]
    #[staticmethod]
    pub fn from_standard(
        py: Python,
        standard: StandardGate,
        qubits: Option<Bound<PyAny>>,
        params: SmallVec<[Param; 3]>,
        label: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            operation: standard.into(),
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: PyTuple::empty(py).unbind(),
            params,
            label: label.map(Box::new),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })
    }

    /// Returns a shallow copy.
    ///
    /// Returns:
    ///     CircuitInstruction: The shallow copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// The logical operation that this instruction represents an execution of.
    #[getter]
    pub fn get_operation(&self, py: Python) -> PyResult<PyObject> {
        // This doesn't use `get_or_init` because a) the initialiser is fallible and
        // `get_or_try_init` isn't stable, and b) the initialiser can yield to the Python
        // interpreter, which might suspend the thread and allow another to inadvertantly attempt to
        // re-enter the cache setter, which isn't safe.

        #[cfg(feature = "cache_pygates")]
        {
            if let Some(cached_op) = self.py_op.get() {
                return Ok(cached_op.clone_ref(py));
            }
        }

        let out = match self.operation.view() {
            OperationRef::StandardGate(standard) => standard
                .create_py_op(
                    py,
                    Some(&self.params),
                    self.label.as_ref().map(|x| x.as_str()),
                )?
                .into_any(),
            OperationRef::StandardInstruction(instruction) => instruction
                .create_py_op(
                    py,
                    Some(&self.params),
                    self.label.as_ref().map(|x| x.as_str()),
                )?
                .into_any(),
            OperationRef::Gate(gate) => gate.gate.clone_ref(py),
            OperationRef::Instruction(instruction) => instruction.instruction.clone_ref(py),
            OperationRef::Operation(operation) => operation.operation.clone_ref(py),
            OperationRef::Unitary(unitary) => unitary
                .create_py_op(py, self.label.as_ref().map(|x| x.as_str()))?
                .into_any(),
        };

        #[cfg(feature = "cache_pygates")]
        {
            self.py_op.get_or_init(|| out.clone_ref(py));
        }

        Ok(out)
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self) -> &str {
        self.operation.name()
    }

    #[getter]
    fn get_params(&self) -> &[Param] {
        self.params.as_slice()
    }

    #[getter]
    fn matrix<'py>(&'py self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self.operation.view().matrix(&self.params);
        matrix.map(move |mat| mat.into_pyarray(py))
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.as_str())
    }

    /// Is the :class:`.Operation` contained in this instruction a Qiskit standard gate?
    pub fn is_standard_gate(&self) -> bool {
        self.operation.try_standard_gate().is_some()
    }

    /// Is the :class:`.Operation` contained in this instruction a subclass of
    /// :class:`.ControlledGate`?
    pub fn is_controlled_gate(&self, py: Python) -> PyResult<bool> {
        match self.operation.view() {
            OperationRef::StandardGate(standard) => Ok(standard.num_ctrl_qubits() != 0),
            OperationRef::Gate(gate) => gate
                .gate
                .bind(py)
                .is_instance(CONTROLLED_GATE.get_bound(py)),
            _ => Ok(false),
        }
    }

    /// Is the :class:`.Operation` contained in this node a directive?
    pub fn is_directive(&self) -> bool {
        self.operation.directive()
    }

    /// Is the :class:`.Operation` contained in this instruction a control-flow operation (i.e. an
    /// instance of :class:`.ControlFlowOp`)?
    pub fn is_control_flow(&self) -> bool {
        self.operation.control_flow()
    }

    /// Does this instruction contain any :class:`.ParameterExpression` parameters?
    pub fn is_parameterized(&self) -> bool {
        self.params
            .iter()
            .any(|x| matches!(x, Param::ParameterExpression(_)))
    }

    /// Creates a shallow copy with the given fields replaced.
    ///
    /// Returns:
    ///     CircuitInstruction: A new instance with the given fields replaced.
    #[pyo3(signature=(operation=None, qubits=None, clbits=None, params=None))]
    pub fn replace(
        &self,
        py: Python,
        operation: Option<&Bound<PyAny>>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
        params: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let qubits = match qubits {
            None => self.qubits.clone_ref(py),
            Some(qubits) => as_tuple(py, Some(qubits))?.unbind(),
        };
        let clbits = match clbits {
            None => self.clbits.clone_ref(py),
            Some(clbits) => as_tuple(py, Some(clbits))?.unbind(),
        };
        let params = params
            .map(|params| params.extract::<SmallVec<[Param; 3]>>())
            .transpose()?;

        if let Some(operation) = operation {
            let op_parts = operation.extract::<OperationFromPython>()?;
            Ok(Self {
                operation: op_parts.operation,
                qubits,
                clbits,
                params: params.unwrap_or(op_parts.params),
                label: op_parts.label,
                #[cfg(feature = "cache_pygates")]
                py_op: operation.clone().unbind().into(),
            })
        } else {
            Ok(Self {
                operation: self.operation.clone(),
                qubits,
                clbits,
                params: params.unwrap_or_else(|| self.params.clone()),
                label: self.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: self.py_op.clone(),
            })
        }
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyResult<PyObject> {
        (
            self.get_operation(py)?,
            self.qubits.bind(py),
            self.clbits.bind(py),
        )
            .into_py_any(py)
    }

    pub fn __repr__(self_: &Bound<Self>, py: Python<'_>) -> PyResult<String> {
        let type_name = self_.get_type().qualname()?;
        let r = self_.try_borrow()?;
        Ok(format!(
            "{}(operation={}, qubits={}, clbits={})",
            type_name,
            r.get_operation(py)?.bind(py).repr()?,
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
    pub fn _legacy_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.get_operation(py)?,
                self.qubits.bind(py).to_list().into(),
                self.clbits.bind(py).to_list().into(),
            ],
        )
    }

    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        self._legacy_format(py)?
            .as_any()
            .get_item(key)?
            .into_py_any(py)
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        self._legacy_format(py)?
            .as_any()
            .try_iter()?
            .into_py_any(py)
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
        fn params_eq(py: Python, left: &[Param], right: &[Param]) -> PyResult<bool> {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (left, right) in left.iter().zip(right) {
                let eq = match left {
                    Param::Float(left) => match right {
                        Param::Float(right) => left == right,
                        Param::ParameterExpression(right) | Param::Obj(right) => {
                            right.bind(py).eq(left)?
                        }
                    },
                    Param::ParameterExpression(left) | Param::Obj(left) => match right {
                        Param::Float(right) => left.bind(py).eq(right)?,
                        Param::ParameterExpression(right) | Param::Obj(right) => {
                            left.bind(py).eq(right)?
                        }
                    },
                };
                if !eq {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        fn eq(
            py: Python<'_>,
            self_: &Bound<CircuitInstruction>,
            other: &Bound<PyAny>,
        ) -> PyResult<Option<bool>> {
            if self_.is(other) {
                return Ok(Some(true));
            }

            let self_ = self_.try_borrow()?;

            if other.is_instance_of::<PyTuple>() {
                return Ok(Some(self_._legacy_format(py)?.eq(other)?));
            }
            let Ok(other) = other.downcast::<CircuitInstruction>() else {
                return Ok(None);
            };
            let other = other.try_borrow()?;

            Ok(Some(
                self_.qubits.bind(py).eq(other.qubits.bind(py))?
                    && self_.clbits.bind(py).eq(other.clbits.bind(py))?
                    && self_.operation.py_eq(py, &other.operation)?
                    && (self_.operation.try_standard_gate().is_none()
                        || params_eq(py, &self_.params, &other.params)?),
            ))
        }

        Ok(match op {
            CompareOp::Eq => match eq(py, self_, other)? {
                Some(res) => PyBool::new(py, res).to_owned().into_any().unbind(),
                None => py.NotImplemented(),
            },
            CompareOp::Ne => match eq(py, self_, other)? {
                Some(res) => PyBool::new(py, !res).to_owned().into_any().unbind(),
                None => py.NotImplemented(),
            },
            _ => py.NotImplemented(),
        })
    }
}

/// A container struct that contains the conversion from some `Operation` subclass input, on its way
/// to becoming a `PackedInstruction`.
///
/// This is the primary way of converting an incoming `Gate` / `Instruction` / `Operation` from
/// Python space into Rust-space data.  A typical access pattern is:
///
/// ```rust
/// #[pyfunction]
/// fn accepts_op_from_python(ob: &Bound<PyAny>) -> PyResult<()> {
///     let py_op = ob.extract::<OperationFromPython>()?;
///     // ... use `py_op.operation`, `py_op.params`, etc.
///     Ok(())
/// }
/// ```
///
/// though you can also accept `ob: OperationFromPython` directly, if you don't also need a handle
/// to the Python object that it came from.  The handle is useful for the Python-operation caching.
#[derive(Debug)]
pub struct OperationFromPython {
    pub operation: PackedOperation,
    pub params: SmallVec<[Param; 3]>,
    pub label: Option<Box<String>>,
}

impl<'py> FromPyObject<'py> for OperationFromPython {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let ob_type = ob
            .getattr(intern!(py, "base_class"))
            .ok()
            .map(|base| base.downcast_into::<PyType>())
            .transpose()?
            .unwrap_or_else(|| ob.get_type());

        let extract_params = || {
            ob.getattr(intern!(py, "params"))
                .ok()
                .map(|params| params.extract())
                .transpose()
                .map(|params| params.unwrap_or_default())
        };

        let extract_params_no_coerce = || {
            ob.getattr(intern!(py, "params"))
                .ok()
                .map(|params| {
                    params
                        .try_iter()?
                        .map(|p| Param::extract_no_coerce(&p?))
                        .collect()
                })
                .transpose()
                .map(|params| params.unwrap_or_default())
        };

        let extract_label = || -> PyResult<Option<Box<String>>> {
            let raw = ob.getattr(intern!(py, "label"))?;
            Ok(raw.extract::<Option<String>>()?.map(Box::new))
        };

        'standard_gate: {
            // Our Python standard gates have a `_standard_gate` field at the class level so we can
            // quickly identify them here without an `isinstance` check.
            let Some(standard) = ob_type
                .getattr(intern!(py, "_standard_gate"))
                .and_then(|standard| standard.extract::<StandardGate>())
                .ok()
            else {
                break 'standard_gate;
            };

            // If the instruction is a controlled gate with a not-all-ones control state, it doesn't
            // fit our definition of standard.  We abuse the fact that we know our standard-gate
            // mapping to avoid an `isinstance` check on `ControlledGate` - a standard gate has
            // nonzero `num_ctrl_qubits` iff it is a `ControlledGate`.
            //
            // `ControlledGate` also has a `base_gate` attribute related to its historical
            // implementation, which technically allows mutations from Python space.  The only
            // mutation of a standard gate's `base_gate` that wouldn't have already broken the
            // Python-space data model is setting a label, so we just catch that case and default
            // back to non-standard-gate handling in that case.
            if standard.num_ctrl_qubits() != 0
                && ((ob.getattr(intern!(py, "ctrl_state"))?.extract::<usize>()?
                    != (1 << standard.num_ctrl_qubits()) - 1)
                    || !ob
                        .getattr(intern!(py, "base_gate"))?
                        .getattr(intern!(py, "label"))?
                        .is_none())
            {
                break 'standard_gate;
            }
            return Ok(OperationFromPython {
                operation: PackedOperation::from_standard_gate(standard),
                params: extract_params()?,
                label: extract_label()?,
            });
        }
        'standard_instr: {
            // Our Python standard instructions have a `_standard_instruction_type` field at the
            // class level so we can quickly identify them here without an `isinstance` check.
            // Once we know the type, we query the object for any type-specific fields we need to
            // read (e.g. a Barrier's number of qubits) to build the Rust representation.
            let Some(standard_type) = ob_type
                .getattr(intern!(py, "_standard_instruction_type"))
                .and_then(|standard| standard.extract::<StandardInstructionType>())
                .ok()
            else {
                break 'standard_instr;
            };
            let standard = match standard_type {
                StandardInstructionType::Barrier => {
                    let num_qubits = ob.getattr(intern!(py, "num_qubits"))?.extract()?;
                    StandardInstruction::Barrier(num_qubits)
                }
                StandardInstructionType::Delay => {
                    let unit = ob.getattr(intern!(py, "unit"))?.extract()?;
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_standard_instruction(
                            StandardInstruction::Delay(unit),
                        ),
                        // If the delay's duration is a Python int, we preserve it rather than
                        // coercing it to a float (e.g. when unit is 'dt').
                        params: extract_params_no_coerce()?,
                        label: extract_label()?,
                    });
                }
                StandardInstructionType::Measure => StandardInstruction::Measure,
                StandardInstructionType::Reset => StandardInstruction::Reset,
            };
            return Ok(OperationFromPython {
                operation: PackedOperation::from_standard_instruction(standard),
                params: extract_params()?,
                label: extract_label()?,
            });
        }

        // We need to check by name here to avoid a circular import during initial loading
        if ob.getattr(intern!(py, "name"))?.extract::<String>()? == "unitary" {
            let params = extract_params()?;
            if let Some(Param::Obj(data)) = params.first() {
                let py_matrix: PyReadonlyArray2<Complex64> = data.extract(py)?;
                let matrix: Option<MatrixView2<Complex64>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::OneQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: SmallVec::new(),
                        label: extract_label()?,
                    });
                }
                let matrix: Option<MatrixView4<Complex64>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::TwoQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: SmallVec::new(),
                        label: extract_label()?,
                    });
                } else {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::NDArray(py_matrix.as_array().to_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: SmallVec::new(),
                        label: extract_label()?,
                    });
                };
            }
        }

        if ob_type.is_subclass(GATE.get_bound(py))? {
            let params = extract_params()?;
            let gate = Box::new(PyGate {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: 0,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                gate: ob.clone().unbind(),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_gate(gate),
                params,
                label: extract_label()?,
            });
        }
        if ob_type.is_subclass(INSTRUCTION.get_bound(py))? {
            let params = extract_params()?;
            let instruction = Box::new(PyInstruction {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                control_flow: ob.is_instance(CONTROL_FLOW_OP.get_bound(py))?,
                instruction: ob.clone().unbind(),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_instruction(instruction),
                params,
                label: extract_label()?,
            });
        }
        if ob_type.is_subclass(OPERATION.get_bound(py))? {
            let params = extract_params()?;
            let operation = Box::new(PyOperation {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                operation: ob.clone().unbind(),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_operation(operation),
                params,
                label: None,
            });
        }
        Err(PyTypeError::new_err(format!("invalid input: {}", ob)))
    }
}

/// Convert a sequence-like Python object to a tuple.
fn as_tuple<'py>(py: Python<'py>, seq: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> {
    let Some(seq) = seq else {
        return Ok(PyTuple::empty(py));
    };
    if seq.is_instance_of::<PyTuple>() {
        Ok(seq.downcast_into_exact::<PyTuple>()?)
    } else if seq.is_instance_of::<PyList>() {
        Ok(seq.downcast_exact::<PyList>()?.to_tuple())
    } else {
        // New tuple from iterable.
        PyTuple::new(
            py,
            seq.try_iter()?
                .map(|o| Ok(o?.unbind()))
                .collect::<PyResult<Vec<PyObject>>>()?,
        )
    }
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
                    " since Qiskit 1.2, and will be removed in Qiskit 3.0.",
                    " Instead, use the `operation`, `qubits` and `clbits` named attributes."
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            // Stack level.  Compared to Python-space calls to `warn`, this is unusually low
            // beacuse all our internal call structure is now Rust-space and invisible to Python.
            1,
        ))
        .map(|_| ())
}
