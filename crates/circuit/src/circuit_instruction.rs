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

use itertools::Itertools;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyDeprecationWarning, PyTypeError};
use pyo3::prelude::*;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use pyo3::types::{IntoPyDict, PyBool, PyDict, PyList, PyTuple, PyType};
use pyo3::IntoPyObjectExt;
use pyo3::{intern, PyObject, PyResult};

use crate::duration::Duration;
use crate::imports::{
    BARRIER, BOX_OP, BREAK_LOOP_OP, CONTINUE_LOOP_OP, CONTROLLED_GATE, DELAY, FOR_LOOP_OP, GATE,
    IF_ELSE_OP, INSTRUCTION, MEASURE, OPERATION, RESET, SWITCH_CASE_OP, UNITARY_GATE,
    WARNINGS_WARN, WHILE_LOOP_OP,
};
use crate::operations::{
    ArrayType, ControlFlow, ControlFlowType, ControlFlowView, InstructionView, Operation,
    OperationRef, Param, Parameters, PyGate, PyInstruction, PyOperation, StandardGate,
    StandardGateView, StandardInstruction, StandardInstructionType, StandardInstructionView,
    UnitaryGate, UnitaryGateView,
};
use crate::packed_instruction::PackedOperation;
use nalgebra::{Dyn, MatrixView2, MatrixView4};
use num_complex::Complex64;
use smallvec::SmallVec;

/// Implemented for various instruction-like reference types.
///
/// Provides ergonomic views of an underlying instruction.
pub trait IntoInstructionView<'a> {
    /// The type of inner circuits contained within this instruction's views.
    type Block;

    /// Returns a view of the operation.
    fn view_operation(self) -> OperationRef<'a>;

    /// Returns a view of this instruction as a standard gate, if applicable.
    fn try_view_standard_gate(self) -> Option<StandardGateView<'a>>;

    /// Returns a view of this instruction as a standard instruction, if applicable.
    fn try_view_standard_instruction(self) -> Option<StandardInstructionView<'a>>;

    /// Returns a view of this instruction as a control flow instruction, if applicable.
    fn try_view_control_flow(self) -> Option<ControlFlowView<'a, Self::Block>>;

    /// Returns the old-style [Param] sequence, unless this is a control
    /// flow instruction.
    fn try_legacy_params(self) -> Option<&'a [Param]>;

    /// Returns an immutable ergonomic view of this instruction.
    #[inline]
    fn view(self) -> InstructionView<'a, Self::Block>
    where
        Self: Copy + Sized,
    {
        match self.view_operation() {
            OperationRef::ControlFlow(_) => {
                InstructionView::ControlFlow(self.try_view_control_flow().unwrap())
            }
            OperationRef::StandardGate(_) => {
                InstructionView::StandardGate(self.try_view_standard_gate().unwrap())
            }
            OperationRef::StandardInstruction(_) => {
                InstructionView::StandardInstruction(self.try_view_standard_instruction().unwrap())
            }
            OperationRef::Gate(g) => InstructionView::Gate(g),
            OperationRef::Instruction(i) => InstructionView::Instruction(i),
            OperationRef::Operation(o) => InstructionView::Operation(o),
            OperationRef::Unitary(u) => InstructionView::Unitary(UnitaryGateView(u)),
        }
    }
}

/// Represents an instruction that is directly convertible to our Python API
/// instruction type.
pub trait Instruction {
    /// Gets a reference to this instruction's operation.
    fn op(&self) -> OperationRef<'_>;

    /// Get a reference to this instruction's parameter list, if applicable.
    ///
    /// For standard gates without parameters this may be [None] or a
    /// `Some(Parameters::Param(smallvec![]))`.
    fn parameters(&self) -> Option<&Parameters<PyObject>>;

    /// Get the label for this instruction.
    fn label(&self) -> Option<&str>;
}

/// Supports creation of a Python-space representation.
pub trait CreatePythonOperation {
    /// Build a reference to the Python-space operation object (the `Gate`, etc) packed into this
    /// instruction.
    ///
    /// A standard-gate or standard-instruction operation object returned by this function is
    /// disconnected from the containing circuit; updates to its parameters, label, duration, unit
    /// and condition will not be propagated back.
    fn create_py_op(&self, py: Python) -> PyResult<Py<PyAny>>;
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
    pub operation: PackedOperation,
    /// A sequence of the qubits that the operation is applied to.
    #[pyo3(get)]
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    #[pyo3(get)]
    pub clbits: Py<PyTuple>,
    pub params: Option<Parameters<PyObject>>,
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

impl Instruction for CircuitInstruction {
    fn op(&self) -> OperationRef<'_> {
        self.operation.view()
    }

    fn parameters(&self) -> Option<&Parameters<PyObject>> {
        self.params.as_ref()
    }

    fn label(&self) -> Option<&str> {
        self.label()
    }
}

impl<'a, T: Instruction> IntoInstructionView<'a> for &'a T {
    type Block = PyObject;

    fn view_operation(self) -> OperationRef<'a> {
        Instruction::op(self)
    }

    fn try_view_standard_gate(self) -> Option<StandardGateView<'a>> {
        let OperationRef::StandardGate(gate) = self.op() else {
            return None;
        };
        let params = match self.parameters() {
            Some(Parameters::Params(params)) => params.as_slice(),
            None => &[],
            _ => panic!("invalid standard gate parameters"),
        };
        Some(StandardGateView(gate, params))
    }

    fn try_view_standard_instruction(self) -> Option<StandardInstructionView<'a>> {
        let OperationRef::StandardInstruction(instruction) = self.op() else {
            return None;
        };
        Some(match instruction {
            StandardInstruction::Barrier(n) => StandardInstructionView::Barrier(n),
            StandardInstruction::Delay(unit) => {
                let Some([duration]) = self.parameters().and_then(|p| match p {
                    Parameters::Params(params) => Some(params.as_slice()),
                    _ => None,
                }) else {
                    panic!("invalid delay parameters");
                };
                StandardInstructionView::Delay { duration, unit }
            }
            StandardInstruction::Measure => StandardInstructionView::Measure,
            StandardInstruction::Reset => StandardInstructionView::Reset,
        })
    }

    fn try_view_control_flow(self) -> Option<ControlFlowView<'a, Self::Block>> {
        let OperationRef::ControlFlow(control) = self.op() else {
            return None;
        };

        Some(match control {
            ControlFlow::Box { duration, .. } => {
                let Some(Parameters::Box { body }) = self.parameters() else {
                    panic!("invalid box parameters");
                };
                ControlFlowView::Box(duration.as_ref(), body)
            }
            ControlFlow::BreakLoop { .. } => ControlFlowView::BreakLoop,
            ControlFlow::ContinueLoop { .. } => ControlFlowView::ContinueLoop,
            ControlFlow::ForLoop { .. } => {
                let Some(Parameters::ForLoop {
                    indexset,
                    loop_param,
                    body,
                }) = self.parameters()
                else {
                    panic!("invalid for loop parameters");
                };
                ControlFlowView::ForLoop {
                    indexset,
                    loop_param: loop_param.as_ref(),
                    body,
                }
            }
            ControlFlow::IfElse { condition, .. } => {
                let Some(Parameters::IfElse {
                    true_body,
                    false_body,
                }) = self.parameters()
                else {
                    panic!("invalid ifelse parameters");
                };
                ControlFlowView::IfElse {
                    condition,
                    true_body,
                    false_body: false_body.as_ref(),
                }
            }
            ControlFlow::Switch {
                target, label_spec, ..
            } => {
                let cases_specifier = label_spec
                    .iter()
                    .zip(
                        self.parameters()
                            .and_then(|p| match p {
                                Parameters::Switch { cases } => Some(cases),
                                _ => None,
                            })
                            .expect("invalid switch parameters"),
                    )
                    .collect();
                ControlFlowView::Switch {
                    target,
                    cases_specifier,
                }
            }
            ControlFlow::While { condition, .. } => {
                let Some(Parameters::While { body }) = self.parameters() else {
                    panic!("invalid while parameters");
                };
                ControlFlowView::While { condition, body }
            }
        })
    }

    fn try_legacy_params(self) -> Option<&'a [Param]> {
        match self.view() {
            InstructionView::StandardGate(_)
            | InstructionView::Gate(_)
            | InstructionView::Operation(_)
            | InstructionView::Unitary(_)
            | InstructionView::Instruction(_) => Some(
                self.parameters()
                    .and_then(|p| match p {
                        Parameters::Params(p) => Some(p.as_slice()),
                        _ => panic!("expected gate parameters"),
                    })
                    .unwrap_or_default(),
            ),
            InstructionView::StandardInstruction(inst) => match inst {
                StandardInstructionView::Delay { duration, .. } => {
                    Some(std::slice::from_ref(&duration))
                }
                _ => Some(&[]),
            },
            _ => panic!(
                "legacy parameters not supported for operation {:?}",
                self.op()
            ),
        }
    }
}

impl<T: Instruction> CreatePythonOperation for T {
    fn create_py_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.view() {
            InstructionView::ControlFlow(cf) => {
                let qubits = self.op().num_qubits();
                let clbits = self.op().num_clbits();
                let kwargs = self
                    .label()
                    .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
                    .transpose()?;
                match cf {
                    ControlFlowView::Box(duration, body) => {
                        let (duration, unit) = match duration {
                            Some(duration) => (Some(duration.py_value(py)?), duration.unit()),
                            None => (None, "dt"),
                        };
                        BOX_OP
                            .get(py)
                            .call(py, (body, duration, unit), kwargs.as_ref())
                    }
                    ControlFlowView::BreakLoop => {
                        BREAK_LOOP_OP
                            .get(py)
                            .call(py, (qubits, clbits), kwargs.as_ref())
                    }
                    ControlFlowView::ContinueLoop => {
                        CONTINUE_LOOP_OP
                            .get(py)
                            .call(py, (qubits, clbits), kwargs.as_ref())
                    }
                    ControlFlowView::ForLoop {
                        indexset,
                        loop_param,
                        body,
                    } => FOR_LOOP_OP.get(py).call(
                        py,
                        (indexset, loop_param.clone(), body),
                        kwargs.as_ref(),
                    ),
                    ControlFlowView::IfElse {
                        condition,
                        true_body,
                        false_body,
                    } => IF_ELSE_OP.get(py).call(
                        py,
                        (condition.clone(), true_body.clone(), false_body.clone()),
                        kwargs.as_ref(),
                    ),
                    ControlFlowView::Switch {
                        target,
                        cases_specifier,
                    } => SWITCH_CASE_OP.get(py).call(
                        py,
                        (
                            target.clone(),
                            cases_specifier
                                .into_iter()
                                .map(|(spec, case)| (spec.clone(), case))
                                .collect_vec(),
                        ),
                        kwargs.as_ref(),
                    ),
                    ControlFlowView::While { condition, body } => {
                        WHILE_LOOP_OP
                            .get(py)
                            .call(py, (condition.clone(), body), kwargs.as_ref())
                    }
                }
            }
            InstructionView::StandardGate(StandardGateView(gate, params)) => {
                gate.create_py_op(py, Some(params), self.label())
            }
            InstructionView::StandardInstruction(instruction) => {
                let kwargs = self
                    .label()
                    .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
                    .transpose()?;
                match instruction {
                    StandardInstructionView::Barrier(num_qubits) => {
                        BARRIER.get(py).call(py, (num_qubits,), kwargs.as_ref())
                    }
                    StandardInstructionView::Delay { duration, unit } => DELAY
                        .get(py)
                        .call1(py, (duration.clone().into_py_any(py)?, unit.to_string())),
                    StandardInstructionView::Measure => {
                        MEASURE.get(py).call(py, (), kwargs.as_ref())
                    }
                    StandardInstructionView::Reset => RESET.get(py).call(py, (), kwargs.as_ref()),
                }
            }
            InstructionView::Gate(gate) => Ok(gate.gate.clone_ref(py)),
            InstructionView::Instruction(instruction) => Ok(instruction.instruction.clone_ref(py)),
            InstructionView::Operation(operation) => Ok(operation.operation.clone_ref(py)),
            InstructionView::Unitary(unitary) => {
                let kwargs = PyDict::new(py);
                if let Some(label) = self.label() {
                    kwargs.set_item(intern!(py, "label"), label.into_py_any(py)?)?;
                }
                let out_array = match &unitary.array {
                    ArrayType::NDArray(arr) => arr.to_pyarray(py),
                    ArrayType::OneQ(arr) => arr.to_pyarray(py),
                    ArrayType::TwoQ(arr) => arr.to_pyarray(py),
                };
                kwargs.set_item(intern!(py, "check_input"), false)?;
                kwargs.set_item(intern!(py, "num_qubits"), self.op().num_qubits())?;
                let gate = UNITARY_GATE
                    .get_bound(py)
                    .call((out_array,), Some(&kwargs))?;
                Ok(gate.unbind())
            }
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
            params: Some(Parameters::Params(params)),
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

        let out = self.create_py_op(py)?;

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
    pub fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        let Some(params) = &self.params else {
            return Ok(PyList::empty(py).into_any().unbind());
        };
        match params {
            Parameters::Params(params) => params.clone().into_py_any(py),
            Parameters::Box { .. } => todo!(),
            Parameters::ForLoop { .. } => todo!(),
            Parameters::IfElse { .. } => todo!(),
            Parameters::Switch { .. } => todo!(),
            Parameters::While { .. } => todo!(),
        }
    }

    #[getter]
    fn matrix<'py>(&'py self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self.view().try_matrix();
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
        self.operation.try_control_flow().is_some()
    }

    /// Does this instruction contain any :class:`.ParameterExpression` parameters?
    pub fn is_parameterized(&self) -> bool {
        let Some(params) = self.params.as_ref() else {
            return false;
        };
        match params {
            Parameters::Params(p) => p.iter().any(|x| matches!(x, Param::ParameterExpression(_))),
            Parameters::Box { .. } => false,
            Parameters::ForLoop { .. } => false,
            Parameters::IfElse { .. } => false,
            Parameters::Switch { .. } => false,
            Parameters::While { .. } => false,
        }
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
        // let params = params
        //     .map(|params| params.extract::<SmallVec<[Param; 3]>>())
        //     .transpose()?;

        if let Some(operation) = operation {
            let op_parts = operation.extract::<OperationFromPython>()?;
            let params = if let Some(params) = params {
                extract_params(op_parts.operation.view(), &params)?
            } else {
                op_parts.params
            };

            Ok(Self {
                operation: op_parts.operation,
                qubits,
                clbits,
                params,
                label: op_parts.label,
                #[cfg(feature = "cache_pygates")]
                py_op: operation.clone().unbind().into(),
            })
        } else {
            let params = if let Some(params) = params {
                extract_params(self.operation.view(), &params)?
            } else {
                self.params.clone()
            };
            Ok(Self {
                operation: self.operation.clone(),
                qubits,
                clbits,
                params,
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
        fn params_eq(
            py: Python,
            left: &Option<Parameters<PyObject>>,
            right: &Option<Parameters<PyObject>>,
        ) -> PyResult<bool> {
            if left.is_none() && right.is_none() {
                return Ok(true);
            }
            let (Some(left), Some(right)) = (left, right) else {
                return Ok(false);
            };

            match (left, right) {
                (Parameters::Params(left), Parameters::Params(right)) => {
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
                (Parameters::Box { body: body_a }, Parameters::Box { body: body_b }) => {
                    body_a.bind(py).eq(body_b)
                }
                (
                    Parameters::ForLoop {
                        indexset: indexset_a,
                        loop_param: loop_param_a,
                        body: body_a,
                    },
                    Parameters::ForLoop {
                        indexset: indexset_b,
                        loop_param: loop_param_b,
                        body: body_b,
                    },
                ) => {
                    let loop_param_eq = || -> PyResult<bool> {
                        match (loop_param_a, loop_param_b) {
                            (Some(loop_param_a), Some(loop_param_b)) => {
                                loop_param_a.bind(py).eq(loop_param_b)
                            }
                            _ => Ok(false),
                        }
                    };
                    Ok(indexset_a == indexset_b
                        && loop_param_eq()?
                        && body_a.bind(py).eq(body_b)?)
                }
                (
                    Parameters::IfElse {
                        true_body: true_body_a,
                        false_body: false_body_a,
                    },
                    Parameters::IfElse {
                        true_body: true_body_b,
                        false_body: false_body_b,
                    },
                ) => {
                    let false_body_eq = || -> PyResult<bool> {
                        match (false_body_a, false_body_b) {
                            (Some(false_body_a), Some(false_body_b)) => {
                                false_body_a.bind(py).eq(false_body_b)
                            }
                            (None, None) => Ok(true),
                            _ => Ok(false),
                        }
                    };
                    Ok(true_body_a.bind(py).eq(true_body_b)? && false_body_eq()?)
                }
                (Parameters::Switch { cases: cases_a }, Parameters::Switch { cases: cases_b }) => {
                    if cases_a.len() != cases_b.len() {
                        return Ok(false);
                    }
                    for (a, b) in cases_a.iter().zip(cases_b) {
                        if !a.bind(py).eq(b)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                (Parameters::While { body: body_a }, Parameters::While { body: body_b }) => {
                    body_a.bind(py).eq(body_b)
                }
                _ => Ok(false),
            }
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
    pub params: Option<Parameters<PyObject>>,
    pub label: Option<Box<String>>,
}

impl Instruction for OperationFromPython {
    fn op(&self) -> OperationRef<'_> {
        self.operation.view()
    }

    fn parameters(&self) -> Option<&Parameters<PyObject>> {
        self.params.as_ref()
    }

    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.as_str())
    }
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

        let get_params = || -> PyResult<Bound<PyAny>> {
            Ok(ob
                .getattr_opt(intern!(py, "params"))?
                .unwrap_or_else(|| PyTuple::empty(py).into_any()))
        };
        // let extract_params = || {
        //     ob.getattr(intern!(py, "params"))
        //         .ok()
        //         .map(|params| params.extract())
        //         .transpose()
        //         .map(|params| params.unwrap_or_default())
        // };

        // let extract_params_no_coerce = || {
        //     ob.getattr(intern!(py, "params"))
        //         .ok()
        //         .map(|params| {
        //             params
        //                 .try_iter()?
        //                 .map(|p| Param::extract_no_coerce(&p?))
        //                 .collect()
        //         })
        //         .transpose()
        //         .map(|params| params.unwrap_or_default())
        // };

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
            let operation = PackedOperation::from_standard_gate(standard);
            let params = extract_params(operation.view(), &get_params()?)?;
            return Ok(OperationFromPython {
                operation,
                params,
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
                    StandardInstruction::Delay(unit)
                }
                StandardInstructionType::Measure => StandardInstruction::Measure,
                StandardInstructionType::Reset => StandardInstruction::Reset,
            };
            let operation = PackedOperation::from_standard_instruction(standard);
            let params = extract_params(operation.view(), &get_params()?)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }
        'control_flow: {
            // Our Python control flow instructions have a `_control_flow_type` field at the
            // class level so we can quickly identify them here without an `isinstance` check.
            // Once we know the type, we query the object for any type-specific fields we need to
            // read to build the Rust representation.
            let Some(control_flow_type) = ob_type
                .getattr(intern!(py, "_control_flow_type"))
                .and_then(|cf| cf.extract::<ControlFlowType>())
                .ok()
            else {
                break 'control_flow;
            };
            let params = get_params()?;
            let control_flow = match control_flow_type {
                ControlFlowType::Box => {
                    let py_duration: Option<Bound<PyAny>> =
                        ob.getattr(intern!(py, "duration"))?.extract()?;
                    let unit: Option<String> = ob.getattr(intern!(py, "unit"))?.extract()?;
                    let duration = if let Some(py_duration) = py_duration {
                        Some(match unit.as_deref().unwrap_or("dt") {
                            "dt" => Duration::dt(py_duration.extract()?),
                            "s" => Duration::s(py_duration.extract()?),
                            "ms" => Duration::ms(py_duration.extract()?),
                            "us" => Duration::us(py_duration.extract()?),
                            "ns" => Duration::ns(py_duration.extract()?),
                            // TODO: handle "ps"
                            _ => panic!("invalid duration"), // TODO: return Err
                        })
                    } else {
                        None
                    };
                    ControlFlow::Box {
                        duration,
                        qubits: ob.getattr("num_qubits")?.extract()?,
                        clbits: ob.getattr("num_clbits")?.extract()?,
                    }
                }
                ControlFlowType::BreakLoop => ControlFlow::BreakLoop {
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                },
                ControlFlowType::ContinueLoop => ControlFlow::ContinueLoop {
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                },
                ControlFlowType::ForLoop => ControlFlow::ForLoop {
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                },
                ControlFlowType::IfElse => ControlFlow::IfElse {
                    condition: ob.getattr(intern!(py, "condition"))?.extract()?,
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                },
                ControlFlowType::SwitchCase => ControlFlow::Switch {
                    target: ob.getattr(intern!(py, "target"))?.extract()?,
                    label_spec: ob.getattr(intern!(py, "_label_spec"))?.extract()?,
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                    cases: params.len()? as u32,
                },
                ControlFlowType::WhileLoop => ControlFlow::While {
                    condition: ob.getattr(intern!(py, "condition"))?.extract()?,
                    qubits: ob.getattr("num_qubits")?.extract()?,
                    clbits: ob.getattr("num_clbits")?.extract()?,
                },
            };
            let operation = PackedOperation::from_control_flow(control_flow.into());
            let params = extract_params(operation.view(), &params)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }

        // We need to check by name here to avoid a circular import during initial loading
        if ob.getattr(intern!(py, "name"))?.extract::<String>()? == "unitary" {
            let params: SmallVec<[Param; 3]> = get_params()?.extract()?;
            if let Some(Param::Obj(data)) = params.first() {
                let py_matrix: PyReadonlyArray2<Complex64> = data.extract(py)?;
                let matrix: Option<MatrixView2<Complex64, Dyn, Dyn>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::OneQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                }
                let matrix: Option<MatrixView4<Complex64, Dyn, Dyn>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::TwoQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                } else {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::NDArray(py_matrix.as_array().to_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                };
            }
        }

        if ob_type.is_subclass(GATE.get_bound(py))? {
            let params = get_params()?;
            let operation = PackedOperation::from_gate(Box::new(PyGate {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: 0,
                params: params.len()? as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                gate: ob.clone().unbind(),
            }));
            let params = extract_params(operation.view(), &params)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }
        if ob_type.is_subclass(INSTRUCTION.get_bound(py))? {
            let params = get_params()?;
            let operation = PackedOperation::from_instruction(Box::new(PyInstruction {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len()? as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                instruction: ob.clone().unbind(),
            }));
            let params = extract_params(operation.view(), &params)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }
        if ob_type.is_subclass(OPERATION.get_bound(py))? {
            let params = get_params()?;
            let operation = PackedOperation::from_operation(Box::new(PyOperation {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len()? as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                operation: ob.clone().unbind(),
            }));
            let params = extract_params(operation.view(), &params)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: None,
            });
        }
        Err(PyTypeError::new_err(format!("invalid input: {}", ob)))
    }
}

/// Extracts a Python-space params list into an optional [Parameters] list, given
/// the corresponding operation reference.
pub fn extract_params(
    op: OperationRef,
    params: &Bound<PyAny>,
) -> PyResult<Option<Parameters<PyObject>>> {
    Ok(match op {
        OperationRef::ControlFlow(cf) => match cf {
            ControlFlow::Box { .. } => Some(Parameters::Box {
                body: params.try_iter()?.next().unwrap()?.unbind(),
            }),
            ControlFlow::BreakLoop { .. } => None,
            ControlFlow::ContinueLoop { .. } => None,
            ControlFlow::ForLoop { .. } => {
                let mut params = params.try_iter()?;
                Some(Parameters::ForLoop {
                    indexset: params.next().unwrap()?.extract()?,
                    loop_param: params
                        .next()
                        .unwrap()?
                        .extract::<Option<Bound<PyAny>>>()?
                        .map(|p| p.unbind()),
                    body: params.next().unwrap()?.unbind(),
                })
            }
            ControlFlow::IfElse { .. } => {
                let mut params = params.try_iter()?;
                Some(Parameters::IfElse {
                    true_body: params.next().unwrap()?.unbind(),
                    false_body: params
                        .next()
                        .unwrap()?
                        .extract::<Option<Bound<PyAny>>>()?
                        .map(|p| p.unbind()),
                })
            }
            ControlFlow::Switch { .. } => {
                let cases: Vec<PyObject> = params
                    .try_iter()?
                    .map(|p| p.map(|p| p.unbind()))
                    .collect::<PyResult<_>>()?;
                Some(Parameters::Switch { cases })
            }
            ControlFlow::While { .. } => Some(Parameters::While {
                body: params.try_iter()?.next().unwrap()?.unbind(),
            }),
        },
        OperationRef::StandardGate(_) => Some(Parameters::Params(params.extract()?)),
        OperationRef::StandardInstruction(i) => {
            match &i {
                StandardInstruction::Barrier(_) => None,
                StandardInstruction::Delay(_) => {
                    // If the delay's duration is a Python int, we preserve it rather than
                    // coercing it to a float (e.g. when unit is 'dt').
                    Some(Parameters::Params(
                        params
                            .try_iter()?
                            .map(|p| Param::extract_no_coerce(&p?))
                            .collect::<PyResult<_>>()?,
                    ))
                }
                StandardInstruction::Measure => None,
                StandardInstruction::Reset => None,
            }
        }
        OperationRef::Unitary(_) => None,
        OperationRef::Gate(_) | OperationRef::Instruction(_) | OperationRef::Operation(_) => {
            Some(Parameters::Params(params.extract()?))
        }
    })
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
