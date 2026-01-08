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

use approx::relative_eq;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::Arc;
use std::{fmt, vec};

use crate::bit::{ClassicalRegister, ShareableClbit};
use crate::circuit_data::CircuitData;
use crate::classical::expr;
use crate::duration::Duration;
use crate::packed_instruction::PackedInstruction;
use crate::parameter::parameter_expression::{
    ParameterExpression, PyParameter, PyParameterExpression,
};
use crate::parameter::symbol_expr::{Symbol, Value};
use crate::{ControlFlowBlocks, Qubit, gate_matrix, impl_intopyobject_for_copy_pyclass, imports};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{Array2, ArrayView2, Dim, ShapeBuilder, array, aview2};
use num_bigint::BigUint;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyFloat, PyList, PyTuple};
use pyo3::{IntoPyObjectExt, Python, intern};

#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(Arc<ParameterExpression>),
    Float(f64),
    Obj(Py<PyAny>),
}

impl<'py> IntoPyObject<'py> for &Param {
    type Target = PyAny; // target type is PyAny to cover f64, Py<PyAny> and PyParameterExpression
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Param::Float(value) => value.into_bound_py_any(py),
            Param::Obj(py_obj) => py_obj.into_bound_py_any(py),
            Param::ParameterExpression(expr) => {
                let py_expr = PyParameterExpression::from(expr.as_ref().clone());
                py_expr.coerce_into_py(py)?.into_bound_py_any(py)
            }
        }
    }
}

impl<'py> IntoPyObject<'py> for Param {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Param {
    type Error = ::std::convert::Infallible;

    fn extract(b: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(if let Ok(py_expr) = b.extract::<PyParameterExpression>() {
            Param::ParameterExpression(Arc::new(py_expr.inner))
        } else if b.is_instance_of::<PyArray1<i32>>() {
            // TODO: remove this branch when we raise the NumPy version to 2.4.
            Param::Obj(b.to_owned().unbind())
        } else if let Ok(val) = b.extract::<f64>() {
            Param::Float(val)
        } else {
            Param::Obj(b.to_owned().unbind())
        })
    }
}

impl Param {
    pub fn eq(&self, other: &Param) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(a == b),
            [Self::Float(a), Self::ParameterExpression(b)] => {
                Ok(&ParameterExpression::from_f64(*a) == b.as_ref())
            }
            [Self::ParameterExpression(a), Self::Float(b)] => {
                Ok(a.as_ref() == &ParameterExpression::from_f64(*b))
            }
            [Self::ParameterExpression(a), Self::ParameterExpression(b)] => Ok(a == b),
            [Self::Obj(_), Self::Float(_)] => Ok(false),
            [Self::Float(_), Self::Obj(_)] => Ok(false),
            [Self::Obj(a), Self::ParameterExpression(b)] => {
                Python::attach(|py| a.bind(py).eq(b.as_ref().clone()))
            }
            [Self::Obj(a), Self::Obj(b)] => Python::attach(|py| a.bind(py).eq(b)),
            [Self::ParameterExpression(a), Self::Obj(b)] => {
                Python::attach(|py| a.as_ref().clone().into_bound_py_any(py)?.eq(b))
            }
        }
    }

    pub fn is_close(&self, other: &Param, max_relative: f64) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(relative_eq!(a, b, max_relative = max_relative)),
            _ => self.eq(other),
        }
    }

    // TODO: Replace `Box<dyn Iterator>` with a custom iterator struct. The
    // Box<dyn Iterator> is an anti-pattern which shouldn't really be needed.
    /// Get an iterator over any `Symbol` instances tracked within this `Param`.
    pub fn iter_parameters(&self) -> PyResult<Box<dyn Iterator<Item = Symbol> + '_>> {
        match self {
            Param::Float(_) => Ok(Box::new(::std::iter::empty())),
            Param::ParameterExpression(expr) => Ok(Box::new(expr.iter_symbols().cloned())),
            Param::Obj(obj) => {
                Python::attach(|py| -> PyResult<Box<dyn Iterator<Item = Symbol>>> {
                    let parameters_attr = intern!(py, "parameters");
                    let obj = obj.bind(py);
                    if obj.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
                        // Note: this code-path is only potentially used by custom user operations
                        let collected: Vec<Symbol> = obj
                            .getattr(parameters_attr)?
                            .try_iter()?
                            .map(|elem| {
                                let elem = elem?;
                                let py_param_bound = elem.cast::<PyParameter>()?;
                                let py_param = py_param_bound.borrow();
                                let symbol = py_param.symbol();
                                Ok(symbol.clone())
                            })
                            .collect::<PyResult<_>>()?;
                        Ok(Box::new(collected.into_iter()))
                    } else {
                        Ok(Box::new(::std::iter::empty()))
                    }
                })
            }
        }
    }

    /// Construct a [Param] from a [ParameterExpression]. Allows type coercion.
    ///
    /// # Arguments
    ///
    /// * expr - The expression to construct the [Param] from.
    /// * coerce_to_float - If `true`, coerce integers and complex (with 0 imaginary part) types to
    ///   [Param::Float]. If `false`, only float types are [Param::Float] and integers and
    ///   complex numbers are represented as [Param::ParameterExpression].
    ///
    /// # Returns
    ///
    /// - `Param` - The [Param] object.
    pub fn from_expr(expr: ParameterExpression, coerce_to_float: bool) -> PyResult<Self> {
        match expr.try_to_value(true) {
            Ok(value) => match value {
                Value::Int(i) => {
                    if coerce_to_float {
                        Ok(Self::Float(i as f64)) // coerce integer to float
                    } else {
                        // Int is not a param type and only comes from Python so dump it in
                        // there until we support DT unit delay from C
                        Python::attach(|py| Ok(Self::Obj(i.into_py_any(py)?)))
                    }
                }
                Value::Real(f) => Ok(Self::Float(f)),
                Value::Complex(c) => {
                    if coerce_to_float && value.is_real() {
                        Ok(Self::Float(c.re))
                    } else {
                        // Complex numbers are only defined in Python custom
                        // objects and aren't valid gate parameters for
                        // anything else so return it as an object
                        Python::attach(|py| Ok(Self::Obj(c.into_py_any(py)?)))
                    }
                }
            },
            Err(_) => Ok(Self::ParameterExpression(Arc::new(expr))),
        }
    }

    /// Extract from a Python object without numeric coercion to float.  The default conversion will
    /// coerce integers into floats, but in things like `assign_parameters`, this is not always
    /// desirable.
    pub fn extract_no_coerce(ob: Borrowed<PyAny>) -> PyResult<Self> {
        Ok(if ob.is_instance_of::<PyFloat>() {
            Param::Float(ob.extract()?)
        } else if let Ok(py_expr) = PyParameterExpression::extract_coerce(ob) {
            // don't get confused by the `coerce` name here -- we promise to not coerce to
            // Param::Float. But if it's an int or complex we need to store it as an Obj.
            if Some(true) == py_expr.inner.is_int() || Some(true) == py_expr.inner.is_complex() {
                Param::Obj(ob.to_owned().unbind())
            } else {
                Param::ParameterExpression(Arc::new(py_expr.inner))
            }
        } else {
            Param::Obj(ob.to_owned().unbind())
        })
    }

    /// Clones the [Param] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone()),
            Param::Float(float) => Param::Float(*float),
            Param::Obj(obj) => Param::Obj(obj.clone_ref(py)),
        }
    }

    pub fn py_deepcopy<'py>(
        &self,
        py: Python<'py>,
        memo: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        match self {
            Param::Float(f) => Ok(Param::Float(*f)),
            _ => imports::DEEPCOPY
                .get_bound(py)
                .call1((self.clone(), memo))?
                .extract()
                // The extraction is infallible.
                .map_err(|x| match x {}),
        }
    }
}

// This impl allows for shared usage between [Param] and &[Param].
// Such blanked impl doesn't exist inherently due to Rust's type system limitations.
// See https://doc.rust-lang.org/std/convert/trait.AsRef.html#reflexivity for more information.
impl AsRef<Param> for Param {
    fn as_ref(&self) -> &Param {
        self
    }
}

// Conveniently converts an f64 into a `Param`.
impl From<f64> for Param {
    fn from(value: f64) -> Self {
        Param::Float(value)
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
#[derive(Debug)]
pub enum OperationRef<'a> {
    ControlFlow(&'a ControlFlowInstruction),
    StandardGate(StandardGate),
    StandardInstruction(StandardInstruction),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(&'a UnitaryGate),
    PauliProductMeasurement(&'a PauliProductMeasurement),
}

impl Operation for OperationRef<'_> {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Self::ControlFlow(op) => op.name(),
            Self::StandardGate(standard) => standard.name(),
            Self::StandardInstruction(instruction) => instruction.name(),
            Self::Gate(gate) => gate.name(),
            Self::Instruction(instruction) => instruction.name(),
            Self::Operation(operation) => operation.name(),
            Self::Unitary(unitary) => unitary.name(),
            Self::PauliProductMeasurement(ppm) => ppm.name(),
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_qubits(),
            Self::StandardGate(standard) => standard.num_qubits(),
            Self::StandardInstruction(instruction) => instruction.num_qubits(),
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(instruction) => instruction.num_qubits(),
            Self::Operation(operation) => operation.num_qubits(),
            Self::Unitary(unitary) => unitary.num_qubits(),
            Self::PauliProductMeasurement(ppm) => ppm.num_qubits(),
        }
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_clbits(),
            Self::StandardGate(standard) => standard.num_clbits(),
            Self::StandardInstruction(instruction) => instruction.num_clbits(),
            Self::Gate(gate) => gate.num_clbits(),
            Self::Instruction(instruction) => instruction.num_clbits(),
            Self::Operation(operation) => operation.num_clbits(),
            Self::Unitary(unitary) => unitary.num_clbits(),
            Self::PauliProductMeasurement(ppm) => ppm.num_clbits(),
        }
    }
    #[inline]
    fn num_params(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_params(),
            Self::StandardGate(standard) => standard.num_params(),
            Self::StandardInstruction(instruction) => instruction.num_params(),
            Self::Gate(gate) => gate.num_params(),
            Self::Instruction(instruction) => instruction.num_params(),
            Self::Operation(operation) => operation.num_params(),
            Self::Unitary(unitary) => unitary.num_params(),
            Self::PauliProductMeasurement(ppm) => ppm.num_params(),
        }
    }
    #[inline]
    fn directive(&self) -> bool {
        match self {
            Self::ControlFlow(op) => op.directive(),
            Self::StandardGate(standard) => standard.directive(),
            Self::StandardInstruction(instruction) => instruction.directive(),
            Self::Gate(gate) => gate.directive(),
            Self::Instruction(instruction) => instruction.directive(),
            Self::Operation(operation) => operation.directive(),
            Self::Unitary(unitary) => unitary.directive(),
            Self::PauliProductMeasurement(ppm) => ppm.directive(),
        }
    }
}

/// Used to tag control flow instructions via the `_control_flow_type` class
/// attribute in the corresponding Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int)]
#[repr(u8)]
pub(crate) enum ControlFlowType {
    Box = 0,
    BreakLoop = 1,
    ContinueLoop = 2,
    ForLoop = 3,
    IfElse = 4,
    SwitchCase = 5,
    WhileLoop = 6,
}

#[derive(Clone, Debug, IntoPyObject, PartialEq)]
pub enum BoxDuration {
    Duration(Duration),
    Expr(expr::Expr),
}

/// A literal Python range extracted to a Rust object.
///
/// This is separate to `PyO3`'s `PyRange` type, since that keeps everything internally safe for
/// subclassing and modification the Python heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PyRange {
    pub start: isize,
    pub stop: isize,
    pub step: NonZero<isize>,
}
impl PyRange {
    pub fn is_empty(&self) -> bool {
        let step = self.step.unsigned_abs().get();
        let diff = self.start.abs_diff(self.stop);
        (self.step.get() > 0 && self.start < self.stop
            || (self.step.get() < 0 && self.start > self.stop))
            && diff >= step
    }
    pub fn len(&self) -> usize {
        let step = self.step.unsigned_abs().get();
        let diff = self.start.abs_diff(self.stop);
        if (self.step.get() > 0 && self.start < self.stop)
            || (self.step.get() < 0 && self.start > self.stop)
        {
            // The `diff-1` is guaranteed safe because the `start < stop` or `start > stop`
            // conditions guarantee that `diff` is at least 1.
            1 + (diff - 1) / step
        } else {
            0
        }
    }
}
impl<'py> IntoPyObject<'py> for PyRange {
    type Target = ::pyo3::types::PyRange;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        ::pyo3::types::PyRange::new_with_step(py, self.start, self.stop, self.step.get())
    }
}
impl<'py> IntoPyObject<'py> for &'_ PyRange {
    type Target = <PyRange as IntoPyObject<'py>>::Target;
    type Output = <PyRange as IntoPyObject<'py>>::Output;
    type Error = <PyRange as IntoPyObject<'py>>::Error;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }
}
impl<'a, 'py> FromPyObject<'a, 'py> for PyRange {
    type Error = PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        use pyo3::types::PyRangeMethods;

        let ob = ob.cast::<pyo3::types::PyRange>()?;
        Ok(Self {
            start: ob.start()?,
            stop: ob.stop()?,
            step: NonZero::new(ob.step()?).expect("Python does not allow zero steps"),
        })
    }
}

/// Possible specifications of the "collection" that a for loop iterates over.
#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef, FromPyObject, PartialEq, Eq)]
pub enum ForCollection {
    /// A literal Python `range` object extracted to Rust.
    PyRange(PyRange),
    /// Some ordered collection of integers.
    List(Vec<usize>),
}
impl ForCollection {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::PyRange(xs) => xs.is_empty(),
            Self::List(xs) => xs.is_empty(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::PyRange(xs) => xs.len(),
            Self::List(xs) => xs.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlFlowInstruction {
    pub control_flow: ControlFlow,
    pub num_qubits: u32,
    pub num_clbits: u32,
}

#[derive(Clone, Debug)]
#[repr(align(8))]
pub enum ControlFlow {
    Box {
        duration: Option<BoxDuration>,
        annotations: Vec<Py<PyAny>>,
    },
    BreakLoop,
    ContinueLoop,
    ForLoop {
        collection: ForCollection,
        loop_param: Option<Symbol>,
    },
    IfElse {
        condition: Condition,
    },
    Switch {
        target: SwitchTarget,
        label_spec: Vec<Vec<CaseSpecifier>>,
        cases: u32,
    },
    While {
        condition: Condition,
    },
}

impl ControlFlowInstruction {
    /// Check if another control flow operations is equivalent to this one.
    ///
    /// This can be removed and [ControlFlowInstruction] can be made to implement [PartialEq]
    /// instead once `annotations` gets moved to the instruction.
    pub fn py_eq(&self, py: Python, other: &ControlFlowInstruction) -> PyResult<bool> {
        if self.num_qubits != other.num_qubits || self.num_clbits != other.num_clbits {
            return Ok(false);
        }
        match &self.control_flow {
            ControlFlow::Box {
                duration: self_duration,
                annotations: self_annotations,
            } => match &other.control_flow {
                ControlFlow::Box {
                    duration: other_duration,
                    annotations: other_annotations,
                } => {
                    if self_duration != other_duration
                        || self_annotations.len() != other_annotations.len()
                    {
                        return Ok(false);
                    }
                    for (a, b) in self_annotations.iter().zip(other_annotations) {
                        if !a.bind(py).eq(b)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            },
            ControlFlow::BreakLoop => match &other.control_flow {
                ControlFlow::BreakLoop => Ok(true),
                _ => Ok(false),
            },
            ControlFlow::ContinueLoop => match &other.control_flow {
                ControlFlow::ContinueLoop => Ok(true),
                _ => Ok(false),
            },
            ControlFlow::ForLoop {
                collection: self_collection,
                loop_param: self_loop_param,
            } => match &other.control_flow {
                ControlFlow::ForLoop {
                    collection: other_collection,
                    loop_param: other_loop_param,
                } => Ok(self_collection == other_collection && self_loop_param == other_loop_param),
                _ => Ok(false),
            },
            ControlFlow::IfElse {
                condition: self_condition,
            } => match &other.control_flow {
                ControlFlow::IfElse {
                    condition: other_condition,
                } => Ok(self_condition == other_condition),
                _ => Ok(false),
            },
            ControlFlow::Switch {
                target: self_target,
                label_spec: self_label_spec,
                cases: self_cases,
            } => match &other.control_flow {
                ControlFlow::Switch {
                    target: other_target,
                    label_spec: other_label_spec,
                    cases: other_cases,
                } => Ok(self_cases == other_cases
                    && self_target == other_target
                    && self_label_spec == other_label_spec),
                _ => Ok(false),
            },
            ControlFlow::While {
                condition: self_condition,
            } => match &other.control_flow {
                ControlFlow::While {
                    condition: other_condition,
                } => Ok(self_condition == other_condition),
                _ => Ok(false),
            },
        }
    }

    pub fn create_py_op(
        &self,
        py: Python,
        blocks: Option<Vec<CircuitData>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut blocks = blocks.into_iter().flatten();
        let kwargs = label
            .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
            .transpose()?;
        match &self.control_flow {
            ControlFlow::Box {
                duration,
                annotations,
            } => {
                let (duration, unit) = match duration {
                    Some(duration) => match duration {
                        BoxDuration::Duration(duration) => {
                            (Some(duration.py_value(py)?), Some(duration.unit()))
                        }
                        BoxDuration::Expr(expr) => {
                            (Some(expr.clone().into_py_any(py)?), Some("expr"))
                        }
                    },
                    None => (None, None),
                };
                imports::BOX_OP.get(py).call1(
                    py,
                    (
                        blocks
                            .next()
                            .expect("box should have a body")
                            .into_py_quantum_circuit(py)?,
                        duration,
                        unit,
                        label,
                        PyTuple::new(py, annotations)?,
                    ),
                )
            }
            ControlFlow::BreakLoop => imports::BREAK_LOOP_OP.get(py).call(
                py,
                (self.num_qubits, self.num_clbits),
                kwargs.as_ref(),
            ),
            ControlFlow::ContinueLoop => imports::CONTINUE_LOOP_OP.get(py).call(
                py,
                (self.num_qubits, self.num_clbits),
                kwargs.as_ref(),
            ),
            ControlFlow::ForLoop {
                collection,
                loop_param,
            } => imports::FOR_LOOP_OP.get(py).call(
                py,
                (
                    collection,
                    loop_param.clone(),
                    blocks
                        .next()
                        .expect("for loop should have a body")
                        .into_py_quantum_circuit(py)?,
                ),
                kwargs.as_ref(),
            ),
            ControlFlow::IfElse { condition } => imports::IF_ELSE_OP.get(py).call(
                py,
                (
                    condition.clone(),
                    blocks
                        .next()
                        .expect("if should have a true body")
                        .into_py_quantum_circuit(py)?,
                    blocks
                        .next()
                        .map(|circuit| circuit.into_py_quantum_circuit(py))
                        .transpose()?,
                ),
                kwargs.as_ref(),
            ),
            ControlFlow::Switch {
                target, label_spec, ..
            } => {
                let cases_specifier: Vec<(Vec<CaseSpecifier>, Py<PyAny>)> = label_spec
                    .iter()
                    .cloned()
                    .zip(blocks)
                    .map(|(cases, body)| {
                        body.into_py_quantum_circuit(py)
                            .map(|ob| (cases, ob.unbind()))
                    })
                    .collect::<PyResult<_>>()?;
                imports::SWITCH_CASE_OP.get(py).call(
                    py,
                    (target.clone(), cases_specifier),
                    kwargs.as_ref(),
                )
            }
            ControlFlow::While { condition, .. } => imports::WHILE_LOOP_OP.get(py).call(
                py,
                (
                    condition.clone(),
                    blocks
                        .next()
                        .expect("while should have a body")
                        .into_py_quantum_circuit(py)?,
                ),
                kwargs.as_ref(),
            ),
        }
    }
}

impl Operation for ControlFlowInstruction {
    fn name(&self) -> &str {
        match &self.control_flow {
            ControlFlow::Box { .. } => "box",
            ControlFlow::BreakLoop => "break_loop",
            ControlFlow::ContinueLoop => "continue_loop",
            ControlFlow::ForLoop { .. } => "for_loop",
            ControlFlow::IfElse { .. } => "if_else",
            ControlFlow::Switch { .. } => "switch_case",
            ControlFlow::While { .. } => "while_loop",
        }
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        self.num_clbits
    }

    fn num_params(&self) -> u32 {
        match &self.control_flow {
            ControlFlow::Box { .. } => 1,
            ControlFlow::BreakLoop => 0,
            ControlFlow::ContinueLoop => 0,
            ControlFlow::ForLoop { .. } => 3,
            ControlFlow::IfElse { .. } => 2,
            ControlFlow::Switch { cases, .. } => *cases,
            ControlFlow::While { .. } => 1,
        }
    }

    fn directive(&self) -> bool {
        false
    }
}

/// An ergonomic view of a control flow operation and its blocks.
#[derive(Clone, Debug)]
pub enum ControlFlowView<'a, T> {
    Box {
        duration: Option<&'a BoxDuration>,
        body: &'a T,
    },
    BreakLoop,
    ContinueLoop,
    ForLoop {
        collection: &'a ForCollection,
        loop_param: Option<&'a Symbol>,
        body: &'a T,
    },
    IfElse {
        condition: &'a Condition,
        true_body: &'a T,
        false_body: Option<&'a T>,
    },
    Switch {
        target: &'a SwitchTarget,
        cases_specifier: Vec<(&'a [CaseSpecifier], &'a T)>,
    },
    While {
        condition: &'a Condition,
        body: &'a T,
    },
}

impl<'a, T> ControlFlowView<'a, T> {
    /// Produce a complete control-flow view object from the given instruction.
    ///
    /// While [CircuitData] and [DAGCircuit] both provide `try_view_control_flow` methods which just
    /// delegate to this internally, this function is useful for a) code de-duplication and b)
    /// finer-grained borrow-check control from within the `impl` blocks of [CircuitData] and
    /// [DAGCircuit].
    ///
    /// Panics or produces invalid results if `inst` and `blocks` aren't from compatible sources
    /// (e.g. the same [CircuitData]).
    pub fn try_from_instruction(
        inst: &'a PackedInstruction,
        blocks: &'a ControlFlowBlocks<T>,
    ) -> Option<Self> {
        let OperationRef::ControlFlow(cf) = inst.op.view() else {
            return None;
        };
        let block_ids = inst.blocks_view();
        let view = match &cf.control_flow {
            ControlFlow::Box {
                duration,
                annotations: _,
            } => Self::Box {
                duration: duration.as_ref(),
                body: &blocks[block_ids[0]],
            },
            ControlFlow::BreakLoop => Self::BreakLoop,
            ControlFlow::ContinueLoop => Self::ContinueLoop,
            ControlFlow::ForLoop {
                collection,
                loop_param,
            } => Self::ForLoop {
                collection,
                loop_param: loop_param.as_ref(),
                body: &blocks[block_ids[0]],
            },
            ControlFlow::IfElse { condition } => Self::IfElse {
                condition,
                true_body: &blocks[block_ids[0]],
                false_body: block_ids.get(1).map(|bid| &blocks[*bid]),
            },
            ControlFlow::Switch {
                target,
                label_spec,
                cases: _,
            } => Self::Switch {
                target,
                cases_specifier: label_spec
                    .iter()
                    .zip(block_ids)
                    .map(|(cases, bid)| (cases.as_slice(), &blocks[*bid]))
                    .collect(),
            },
            ControlFlow::While { condition } => Self::While {
                condition,
                body: &blocks[block_ids[0]],
            },
        };
        Some(view)
    }

    pub fn blocks(&self) -> Vec<&'a T> {
        match self {
            ControlFlowView::Box { body, .. } => vec![*body],
            ControlFlowView::BreakLoop => vec![],
            ControlFlowView::ContinueLoop => vec![],
            ControlFlowView::ForLoop { body, .. } => vec![*body],
            ControlFlowView::IfElse {
                true_body,
                false_body,
                ..
            } => {
                if let Some(false_body) = false_body {
                    vec![*true_body, *false_body]
                } else {
                    vec![*true_body]
                }
            }
            ControlFlowView::Switch {
                cases_specifier, ..
            } => cases_specifier.iter().map(|(_, block)| *block).collect(),
            ControlFlowView::While { body, .. } => vec![*body],
        }
    }
}

/// A control flow operation's condition.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum Condition {
    Bit(ShareableClbit, bool),
    Register(ClassicalRegister, BigUint),
    Expr(expr::Expr),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Condition {
    type Error = <expr::Expr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok((bit, value)) = ob.extract::<(ShareableClbit, Bound<PyAny>)>() {
            Ok(Condition::Bit(bit, value.is_truthy()?))
        } else if let Ok((register, value)) = ob.extract::<(ClassicalRegister, BigUint)>() {
            Ok(Condition::Register(register, value))
        } else {
            Ok(Condition::Expr(ob.extract()?))
        }
    }
}

/// A control flow operation's target.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum SwitchTarget {
    Bit(ShareableClbit),
    Register(ClassicalRegister),
    Expr(expr::Expr),
}

impl<'a, 'py> FromPyObject<'a, 'py> for SwitchTarget {
    type Error = <expr::Expr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(bit) = ob.extract::<ShareableClbit>() {
            Ok(SwitchTarget::Bit(bit))
        } else if let Ok(register) = ob.extract::<ClassicalRegister>() {
            Ok(SwitchTarget::Register(register))
        } else {
            Ok(SwitchTarget::Expr(ob.extract()?))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CaseSpecifier {
    Uint(BigUint),
    Default,
}

impl<'a, 'py> FromPyObject<'a, 'py> for CaseSpecifier {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(i) = ob.extract::<BigUint>() {
            Ok(CaseSpecifier::Uint(i))
        } else if ob.is(imports::SWITCH_CASE_DEFAULT.get_bound(ob.py())) {
            Ok(CaseSpecifier::Default)
        } else {
            Err(PyValueError::new_err("invalid case specifier"))
        }
    }
}

impl<'py> IntoPyObject<'py> for CaseSpecifier {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            CaseSpecifier::Uint(u) => u.into_bound_py_any(py),
            CaseSpecifier::Default => Ok(imports::SWITCH_CASE_DEFAULT.get_bound(py).clone()),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum DelayUnit {
    NS,
    PS,
    US,
    MS,
    S,
    DT,
    EXPR,
}

unsafe impl ::bytemuck::CheckedBitPattern for DelayUnit {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 7
    }
}
unsafe impl ::bytemuck::NoUninit for DelayUnit {}

impl fmt::Display for DelayUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DelayUnit::NS => "ns",
                DelayUnit::PS => "ps",
                DelayUnit::US => "us",
                DelayUnit::MS => "ms",
                DelayUnit::S => "s",
                DelayUnit::DT => "dt",
                DelayUnit::EXPR => "expr",
            }
        )
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DelayUnit {
    type Error = PyErr;

    fn extract(b: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let str: String = b.extract()?;
        Ok(match str.as_str() {
            "ns" => DelayUnit::NS,
            "ps" => DelayUnit::PS,
            "us" => DelayUnit::US,
            "ms" => DelayUnit::MS,
            "s" => DelayUnit::S,
            "dt" => DelayUnit::DT,
            "expr" => DelayUnit::EXPR,
            unknown_unit => {
                return Err(PyValueError::new_err(format!(
                    "Unit '{unknown_unit}' is invalid."
                )));
            }
        })
    }
}

/// An internal type used to further discriminate the payload of a `PackedOperation` when its
/// discriminant is `PackedOperationType::StandardInstruction`.
///
/// This is also used to tag standard instructions via the `_standard_instruction_type` class
/// attribute in the corresponding Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int)]
#[repr(u8)]
pub(crate) enum StandardInstructionType {
    Barrier = 0,
    Delay = 1,
    Measure = 2,
    Reset = 3,
}

unsafe impl ::bytemuck::CheckedBitPattern for StandardInstructionType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 4
    }
}
unsafe impl ::bytemuck::NoUninit for StandardInstructionType {}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
pub enum StandardInstruction {
    Barrier(u32),
    Delay(DelayUnit),
    Measure,
    Reset,
}

// This must be kept up-to-date with `StandardInstruction` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_INSTRUCTION_SIZE: usize = 4;

impl Operation for StandardInstruction {
    fn name(&self) -> &str {
        match self {
            StandardInstruction::Barrier(_) => "barrier",
            StandardInstruction::Delay(_) => "delay",
            StandardInstruction::Measure => "measure",
            StandardInstruction::Reset => "reset",
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(num_qubits) => *num_qubits,
            StandardInstruction::Delay(_) => 1,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 1,
        }
    }

    fn num_clbits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(_) => 0,
            StandardInstruction::Delay(_) => 0,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 0,
        }
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        match self {
            StandardInstruction::Barrier(_) => true,
            StandardInstruction::Delay(_) => false,
            StandardInstruction::Measure => false,
            StandardInstruction::Reset => false,
        }
    }
}

impl StandardInstruction {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut params = params.into_iter().flatten();
        let kwargs = label
            .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
            .transpose()?;
        let out = match self {
            StandardInstruction::Barrier(num_qubits) => imports::BARRIER
                .get_bound(py)
                .call((num_qubits,), kwargs.as_ref())?,
            StandardInstruction::Delay(unit) => {
                let duration = params.next().unwrap();
                imports::DELAY
                    .get_bound(py)
                    .call1((duration.into_py_any(py)?, unit.to_string()))?
            }
            StandardInstruction::Measure => {
                imports::MEASURE.get_bound(py).call((), kwargs.as_ref())?
            }
            StandardInstruction::Reset => imports::RESET.get_bound(py).call((), kwargs.as_ref())?,
        };

        Ok(out.unbind())
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int)]
pub enum StandardGate {
    GlobalPhase = 0,
    H = 1,
    I = 2,
    X = 3,
    Y = 4,
    Z = 5,
    Phase = 6,
    R = 7,
    RX = 8,
    RY = 9,
    RZ = 10,
    S = 11,
    Sdg = 12,
    SX = 13,
    SXdg = 14,
    T = 15,
    Tdg = 16,
    U = 17,
    U1 = 18,
    U2 = 19,
    U3 = 20,
    CH = 21,
    CX = 22,
    CY = 23,
    CZ = 24,
    DCX = 25,
    ECR = 26,
    Swap = 27,
    ISwap = 28,
    CPhase = 29,
    CRX = 30,
    CRY = 31,
    CRZ = 32,
    CS = 33,
    CSdg = 34,
    CSX = 35,
    CU = 36,
    CU1 = 37,
    CU3 = 38,
    RXX = 39,
    RYY = 40,
    RZZ = 41,
    RZX = 42,
    XXMinusYY = 43,
    XXPlusYY = 44,
    CCX = 45,
    CCZ = 46,
    CSwap = 47,
    RCCX = 48,
    C3X = 49,
    C3SX = 50,
    RC3X = 51,
    // Remember to update StandardGate::is_valid_bit_pattern below
    // if you add or remove this enum's variants!
}
impl_intopyobject_for_copy_pyclass!(StandardGate);

unsafe impl ::bytemuck::CheckedBitPattern for StandardGate {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < (STANDARD_GATE_SIZE as u8)
    }
}
unsafe impl ::bytemuck::NoUninit for StandardGate {}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0-9
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10-19
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 20-29
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 30-39
    2, 2, 2, 2, 2, 3, 3, 3, 3, 4, // 40-49
    4, 4, // 50-51
];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    1, 0, 0, 0, 0, 0, 1, 2, 1, 1, // 0-9
    1, 0, 0, 0, 0, 0, 0, 3, 1, 2, // 10-19
    3, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 0, 0, 0, 4, 1, 3, 1, // 30-39
    1, 1, 1, 2, 2, 0, 0, 0, 0, 0, // 40-49
    0, 0, // 50-51
];

static STANDARD_GATE_NUM_CTRL_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10-19
    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 30-39
    0, 0, 0, 0, 0, 2, 2, 1, 0, 3, // 40-49
    3, 0, // 50-51
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "global_phase", // 0
    "h",            // 1
    "id",           // 2
    "x",            // 3
    "y",            // 4
    "z",            // 5
    "p",            // 6
    "r",            // 7
    "rx",           // 8
    "ry",           // 9
    "rz",           // 10
    "s",            // 11
    "sdg",          // 12
    "sx",           // 13
    "sxdg",         // 14
    "t",            // 15
    "tdg",          // 16
    "u",            // 17
    "u1",           // 18
    "u2",           // 19
    "u3",           // 20
    "ch",           // 21
    "cx",           // 22
    "cy",           // 23
    "cz",           // 24
    "dcx",          // 25
    "ecr",          // 26
    "swap",         // 27
    "iswap",        // 28
    "cp",           // 29
    "crx",          // 30
    "cry",          // 31
    "crz",          // 32
    "cs",           // 33
    "csdg",         // 34
    "csx",          // 35
    "cu",           // 36
    "cu1",          // 37
    "cu3",          // 38
    "rxx",          // 39
    "ryy",          // 40
    "rzz",          // 41
    "rzx",          // 42
    "xx_minus_yy",  // 43
    "xx_plus_yy",   // 44
    "ccx",          // 45
    "ccz",          // 46
    "cswap",        // 47
    "rccx",         // 48
    "mcx",          // 49 ("c3x")
    "c3sx",         // 50
    "rcccx",        // 51 ("rc3x")
];

/// Get a slice of all standard gate names.
pub fn get_standard_gate_names() -> &'static [&'static str] {
    &STANDARD_GATE_NAME
}

impl StandardGate {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = imports::get_std_gate_class(py, *self)?;
        let args = match params {
            None => PyTuple::empty(py),
            Some(params) => {
                PyTuple::new(py, params.into_iter().map(|x| x.into_pyobject(py).unwrap()))?
            }
        };
        if let Some(label) = label {
            let kwargs = [("label", label.into_pyobject(py)?)].into_py_dict(py)?;
            gate_class.call(py, args, Some(&kwargs))
        } else {
            gate_class.call(py, args, None)
        }
    }

    pub fn num_ctrl_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_CTRL_QUBITS[*self as usize]
    }

    pub fn inverse(&self, params: &[Param]) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        match self {
            Self::GlobalPhase => Some((
                Self::GlobalPhase,
                smallvec![multiply_param(&params[0], -1.0)],
            )),
            Self::H => Some((Self::H, smallvec![])),
            Self::I => Some((Self::I, smallvec![])),
            Self::X => Some((Self::X, smallvec![])),
            Self::Y => Some((Self::Y, smallvec![])),
            Self::Z => Some((Self::Z, smallvec![])),
            Self::Phase => Some((Self::Phase, smallvec![multiply_param(&params[0], -1.0)])),
            Self::R => Some((
                Self::R,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::RX => Some((Self::RX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RY => Some((Self::RY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZ => Some((Self::RZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::S => Some((Self::Sdg, smallvec![])),
            Self::Sdg => Some((Self::S, smallvec![])),
            Self::SX => Some((Self::SXdg, smallvec![])),
            Self::SXdg => Some((Self::SX, smallvec![])),
            Self::T => Some((Self::Tdg, smallvec![])),
            Self::Tdg => Some((Self::T, smallvec![])),
            Self::U => Some((
                Self::U,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::U1 => Some((Self::U1, smallvec![multiply_param(&params[0], -1.0)])),
            Self::U2 => Some((
                Self::U2,
                smallvec![
                    add_param(&multiply_param(&params[1], -1.0), -PI),
                    add_param(&multiply_param(&params[0], -1.0), PI),
                ],
            )),
            Self::U3 => Some((
                Self::U3,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::CH => Some((Self::CH, smallvec![])),
            Self::CX => Some((Self::CX, smallvec![])),
            Self::CY => Some((Self::CY, smallvec![])),
            Self::CZ => Some((Self::CZ, smallvec![])),
            Self::DCX => None, // the inverse in not a StandardGate
            Self::ECR => Some((Self::ECR, smallvec![])),
            Self::Swap => Some((Self::Swap, smallvec![])),
            Self::ISwap => None, // the inverse in not a StandardGate
            Self::CPhase => Some((Self::CPhase, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRX => Some((Self::CRX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRY => Some((Self::CRY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRZ => Some((Self::CRZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CS => Some((Self::CSdg, smallvec![])),
            Self::CSdg => Some((Self::CS, smallvec![])),
            Self::CSX => None, // the inverse in not a StandardGate
            Self::CU => Some((
                Self::CU,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                    multiply_param(&params[3], -1.0),
                ],
            )),
            Self::CU1 => Some((Self::CU1, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CU3 => Some((
                Self::CU3,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::RXX => Some((Self::RXX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RYY => Some((Self::RYY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZZ => Some((Self::RZZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZX => Some((Self::RZX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::XXMinusYY => Some((
                Self::XXMinusYY,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::XXPlusYY => Some((
                Self::XXPlusYY,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::CCX => Some((Self::CCX, smallvec![])),
            Self::CCZ => Some((Self::CCZ, smallvec![])),
            Self::CSwap => Some((Self::CSwap, smallvec![])),
            Self::RCCX => None, // the inverse in not a StandardGate
            Self::C3X => Some((Self::C3X, smallvec![])),
            Self::C3SX => None, // the inverse in not a StandardGate
            Self::RC3X => None, // the inverse in not a StandardGate
        }
    }

    pub fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::GlobalPhase => match params {
                [Param::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            Self::H => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            Self::I => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            Self::X => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            Self::Y => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            Self::Z => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            Self::Phase => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::R => match params {
                [Param::Float(theta), Param::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            Self::RX => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RY => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RZ => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::S => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            Self::Sdg => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
            Self::SX => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            Self::SXdg => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
                _ => None,
            },
            Self::T => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            Self::Tdg => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            Self::U => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U1 => match params[0] {
                Param::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            Self::U2 => match params {
                [Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::CH => match params {
                [] => Some(aview2(&gate_matrix::CH_GATE).to_owned()),
                _ => None,
            },
            Self::CX => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            Self::CY => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            Self::CZ => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            Self::DCX => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            Self::ECR => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            Self::Swap => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            Self::ISwap => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            Self::CPhase => match params {
                [Param::Float(lam)] => Some(aview2(&gate_matrix::cp_gate(*lam)).to_owned()),
                _ => None,
            },
            Self::CRX => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRY => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRZ => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CS => match params {
                [] => Some(aview2(&gate_matrix::CS_GATE).to_owned()),
                _ => None,
            },
            Self::CSdg => match params {
                [] => Some(aview2(&gate_matrix::CSDG_GATE).to_owned()),
                _ => None,
            },
            Self::CSX => match params {
                [] => Some(aview2(&gate_matrix::CSX_GATE).to_owned()),
                _ => None,
            },
            Self::CU => match params {
                [
                    Param::Float(theta),
                    Param::Float(phi),
                    Param::Float(lam),
                    Param::Float(gamma),
                ] => Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned()),
                _ => None,
            },
            Self::CU1 => match params[0] {
                Param::Float(lam) => Some(aview2(&gate_matrix::cu1_gate(lam)).to_owned()),
                _ => None,
            },
            Self::CU3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::cu3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::RXX => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rxx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RYY => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::ryy_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZZ => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzz_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZX => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::XXMinusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::XXPlusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::CCX => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            Self::CCZ => match params {
                [] => Some(aview2(&gate_matrix::CCZ_GATE).to_owned()),
                _ => None,
            },
            Self::CSwap => match params {
                [] => Some(aview2(&gate_matrix::CSWAP_GATE).to_owned()),
                _ => None,
            },
            Self::RCCX => match params {
                [] => Some(aview2(&gate_matrix::RCCX_GATE).to_owned()),
                _ => None,
            },
            Self::C3X => match params {
                [] => Some(aview2(&gate_matrix::C3X_GATE).to_owned()),
                _ => None,
            },
            Self::C3SX => match params {
                [] => Some(aview2(&gate_matrix::C3SX_GATE).to_owned()),
                _ => None,
            },
            Self::RC3X => match params {
                [] => Some(aview2(&gate_matrix::RC3X_GATE).to_owned()),
                _ => None,
            },
        }
    }

    pub fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::GlobalPhase => Some(
                CircuitData::from_standard_gates(0, [], params[0].clone())
                    .expect("Unexpected Qiskit python bug"),
            ),
            Self::H => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI / 2.), FLOAT_ZERO, Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::I => None,
            Self::X => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI), FLOAT_ZERO, Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Y => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![
                            Param::Float(PI),
                            Param::Float(PI / 2.),
                            Param::Float(PI / 2.),
                        ],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::Z => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Phase => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![FLOAT_ZERO, FLOAT_ZERO, params[0].clone()],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::R => {
                let theta_expr = clone_param(&params[0]);
                let phi_expr1 = add_param(&params[1], -PI / 2.);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(Self::U, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RX => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(
                            Self::R,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RY => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(
                            Self::R,
                            smallvec![theta.clone(), Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZ => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(Self::Phase, smallvec![theta.clone()], smallvec![Qubit(0)])],
                        multiply_param(theta, -0.5),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::S => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI / 2.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Sdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(-PI / 2.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::SX => Some(
                CircuitData::from_standard_gates(
                    1,
                    [
                        (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::SXdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(-PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::T => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI / 4.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Tdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(-PI / 4.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U => None,
            Self::U1 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        params.iter().cloned().collect(),
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U2 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U3 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        params.iter().cloned().collect(),
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CH => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::S, smallvec![], q1.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Tdg, smallvec![], q1.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::Sdg, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }

            Self::CX => None,
            Self::CY => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::S, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CZ => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::DCX => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::ECR => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::SX, smallvec![], smallvec![Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::X, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(-PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Swap => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::ISwap => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::S, smallvec![], smallvec![Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CPhase => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Phase, smallvec![multiply_param(&params[0], 0.5)], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (
                                Self::Phase,
                                smallvec![multiply_param(&params[0], -0.5)],
                                q1.clone(),
                            ),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Phase, smallvec![multiply_param(&params[0], 0.5)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CRX => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::S, smallvec![], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::Sdg, smallvec![], smallvec![Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CRY => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CRZ => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CS => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::T, smallvec![], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::Tdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::T, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CSdg => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Tdg, smallvec![], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Tdg, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CSX => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CS, smallvec![], q0_1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CU => {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], 0.5),
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], -0.5),
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5),
                    multiply_param(&params[2], -0.5),
                );
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::Phase,
                                smallvec![params[3].clone()],
                                smallvec![Qubit(0)],
                            ),
                            (Self::Phase, smallvec![param_second_p], smallvec![Qubit(0)]),
                            (Self::Phase, smallvec![param_third_p], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], -0.5),
                                    FLOAT_ZERO,
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CU1 => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], 0.5)],
                            smallvec![Qubit(0)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], -0.5)],
                            smallvec![Qubit(1)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], 0.5)],
                            smallvec![Qubit(1)],
                        ),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CU3 => {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], 0.5),
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], -0.5),
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5),
                    multiply_param(&params[2], -0.5),
                );
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Phase, smallvec![param_first_u1], smallvec![Qubit(0)]),
                            (Self::Phase, smallvec![param_second_u1], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], -0.5),
                                    FLOAT_ZERO,
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RXX => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q0.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::H, smallvec![], q1),
                            (Self::H, smallvec![], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::SXdg, smallvec![], q0.clone()),
                            (Self::SXdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::SX, smallvec![], q0),
                            (Self::SX, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZZ => {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1),
                            (Self::CX, smallvec![], q0_q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZX => {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::XXMinusYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::RZ, smallvec![multiply_param(beta, -1.0)], q1.clone()),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::SX, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::RY, smallvec![multiply_param(theta, 0.5)], q0.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::SXdg, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q0),
                            (Self::RZ, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::XXPlusYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::RZ, smallvec![beta.clone()], q0.clone()),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::SX, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q0.clone()),
                            (Self::CX, smallvec![], q1_0.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q1.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q0.clone()),
                            (Self::CX, smallvec![], q1_0),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::SXdg, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q1),
                            (Self::RZ, smallvec![multiply_param(beta, -1.0)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CCX => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        3,
                        [
                            (Self::H, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2.clone()),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::H, smallvec![], q2),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::T, smallvec![], q0),
                            (Self::Tdg, smallvec![], q1),
                            (Self::CX, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }

            Self::CCZ => Some(
                CircuitData::from_standard_gates(
                    3,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(2)]),
                        (
                            Self::CCX,
                            smallvec![],
                            smallvec![Qubit(0), Qubit(1), Qubit(2)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(2)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CSwap => Some(
                CircuitData::from_standard_gates(
                    3,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                        (
                            Self::CCX,
                            smallvec![],
                            smallvec![Qubit(0), Qubit(1), Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::RCCX => {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        3,
                        [
                            (Self::H, smallvec![], q2.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2.clone()),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::H, smallvec![], q2),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::C3X => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(0)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(1)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(1)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::C3SX => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(0), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(1), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(1), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::RC3X => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
        }
    }

    pub fn matrix_as_static_1q(&self, params: &[Param]) -> Option<[[Complex64; 2]; 2]> {
        match self {
            Self::GlobalPhase => None,
            Self::H => match params {
                [] => Some(gate_matrix::H_GATE),
                _ => None,
            },
            Self::I => match params {
                [] => Some(gate_matrix::ONE_QUBIT_IDENTITY),
                _ => None,
            },
            Self::X => match params {
                [] => Some(gate_matrix::X_GATE),
                _ => None,
            },
            Self::Y => match params {
                [] => Some(gate_matrix::Y_GATE),
                _ => None,
            },
            Self::Z => match params {
                [] => Some(gate_matrix::Z_GATE),
                _ => None,
            },
            Self::Phase => match params {
                [Param::Float(theta)] => Some(gate_matrix::phase_gate(*theta)),
                _ => None,
            },
            Self::R => match params {
                [Param::Float(theta), Param::Float(phi)] => Some(gate_matrix::r_gate(*theta, *phi)),
                _ => None,
            },
            Self::RX => match params {
                [Param::Float(theta)] => Some(gate_matrix::rx_gate(*theta)),
                _ => None,
            },
            Self::RY => match params {
                [Param::Float(theta)] => Some(gate_matrix::ry_gate(*theta)),
                _ => None,
            },
            Self::RZ => match params {
                [Param::Float(theta)] => Some(gate_matrix::rz_gate(*theta)),
                _ => None,
            },
            Self::S => match params {
                [] => Some(gate_matrix::S_GATE),
                _ => None,
            },
            Self::Sdg => match params {
                [] => Some(gate_matrix::SDG_GATE),
                _ => None,
            },
            Self::SX => match params {
                [] => Some(gate_matrix::SX_GATE),
                _ => None,
            },
            Self::SXdg => match params {
                [] => Some(gate_matrix::SXDG_GATE),
                _ => None,
            },
            Self::T => match params {
                [] => Some(gate_matrix::T_GATE),
                _ => None,
            },
            Self::Tdg => match params {
                [] => Some(gate_matrix::TDG_GATE),
                _ => None,
            },
            Self::U => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(gate_matrix::u_gate(*theta, *phi, *lam))
                }
                _ => None,
            },
            Self::U1 => match params[0] {
                Param::Float(val) => Some(gate_matrix::u1_gate(val)),
                _ => None,
            },
            Self::U2 => match params {
                [Param::Float(phi), Param::Float(lam)] => Some(gate_matrix::u2_gate(*phi, *lam)),
                _ => None,
            },
            Self::U3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(gate_matrix::u3_gate(*theta, *phi, *lam))
                }
                _ => None,
            },
            Self::CH => None,
            Self::CX => None,
            Self::CY => None,
            Self::CZ => None,
            Self::DCX => None,
            Self::ECR => None,
            Self::Swap => None,
            Self::ISwap => None,
            Self::CPhase => None,
            Self::CRX => None,
            Self::CRY => None,
            Self::CRZ => None,
            Self::CS => None,
            Self::CSdg => None,
            Self::CSX => None,
            Self::CU => None,
            Self::CU1 => None,
            Self::CU3 => None,
            Self::RXX => None,
            Self::RYY => None,
            Self::RZZ => None,
            Self::RZX => None,
            Self::XXMinusYY => None,
            Self::XXPlusYY => None,
            Self::CCX => None,
            Self::CCZ => None,
            Self::CSwap => None,
            Self::RCCX => None,
            Self::C3X => None,
            Self::C3SX => None,
            Self::RC3X => None,
        }
    }
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix<'py>(
        &self,
        py: Python<'py>,
        params: Vec<Param>,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        self.matrix(&params).map(|x| x.into_pyarray(py))
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<CircuitData> {
        self.definition(&params)
    }

    pub fn _inverse(&self, params: Vec<Param>) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        self.inverse(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_ctrl_qubits(&self) -> u32 {
        self.num_ctrl_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }

    #[getter]
    pub fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits() > 0
    }

    #[getter]
    pub fn get_gate_class(&self, py: Python) -> PyResult<&'static Py<PyAny>> {
        imports::get_std_gate_class(py, *self)
    }

    #[staticmethod]
    pub fn all_gates(py: Python) -> PyResult<Bound<PyList>> {
        PyList::new(
            py,
            (0..STANDARD_GATE_SIZE as u8).map(::bytemuck::checked::cast::<_, Self>),
        )
    }

    pub fn __hash__(&self) -> isize {
        *self as isize
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 52;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }
    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }
    fn directive(&self) -> bool {
        false
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

// Return explicitly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &Param) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone()),
        Param::Obj(_) => unreachable!(),
    }
}

/// Multiply a ``Param`` with a float.
pub fn multiply_param(param: &Param, mult: f64) -> Param {
    match param {
        Param::Float(theta) => Param::Float(theta * mult),
        Param::ParameterExpression(theta) => {
            // safe to unwrap as multiplication with float does not have name conflicts
            Param::ParameterExpression(Arc::new(
                theta.mul(&ParameterExpression::from_f64(mult)).unwrap(),
            ))
        }
        Param::Obj(_) => unreachable!("Unsupported multiplication of a Param::Obj."),
    }
}

/// Multiply two ``Param``s.
pub fn multiply_params(param1: Param, param2: Param) -> Param {
    match (&param1, &param2) {
        (Param::Float(theta), Param::Float(lambda)) => Param::Float(theta * lambda),
        (param, Param::Float(theta)) => multiply_param(param, *theta),
        (Param::Float(theta), param) => multiply_param(param, *theta),
        (Param::ParameterExpression(p1), Param::ParameterExpression(p2)) => {
            // TODO we could properly propagate the error here
            Param::ParameterExpression(Arc::new(p1.mul(p2).expect("Name conflict during mul.")))
        }
        _ => unreachable!("Unsupported multiplication."),
    }
}

pub fn add_param(param: &Param, summand: f64) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            // safe to unwrap as addition with float does not have name conflicts
            Arc::new(theta.add(&ParameterExpression::from_f64(summand)).unwrap()),
        ),
        Param::Obj(_) => unreachable!("Unsupported addition of a Param::Obj."),
    }
}

pub fn radd_param(param1: Param, param2: Param) -> Param {
    match [&param1, &param2] {
        [Param::Float(theta), Param::Float(lambda)] => Param::Float(theta + lambda),
        [Param::Float(theta), Param::ParameterExpression(_lambda)] => add_param(&param2, *theta),
        [Param::ParameterExpression(_theta), Param::Float(lambda)] => add_param(&param1, *lambda),
        [
            Param::ParameterExpression(theta),
            Param::ParameterExpression(lambda),
        ] => {
            // TODO we could properly propagate the error here
            Param::ParameterExpression(Arc::new(
                theta.add(lambda).expect("Name conflict during add."),
            ))
        }
        _ => unreachable!("Unsupported addition."),
    }
}

/// This trait is defined on operation types in the circuit that are defined in Python.
/// It contains the methods for managing the Python aspect
pub trait PythonOperation: Sized {
    /// Copy this operation, including a Python-space deep copy
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self>;

    /// Copy this operation, including a Python-space call to `copy` on the `Operation` subclass.
    fn py_copy(&self, py: Python) -> PyResult<Self>;
}

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyInstruction {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub instruction: Py<PyAny>,
}

impl PythonOperation for PyInstruction {
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let deepcopy = imports::DEEPCOPY.get_bound(py);
        Ok(PyInstruction {
            instruction: deepcopy.call1((&self.instruction, memo))?.unbind(),
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }

    fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        Ok(PyInstruction {
            instruction: self.instruction.call_method0(py, copy_attr)?,
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }

    fn directive(&self) -> bool {
        Python::attach(|py| -> bool {
            match self.instruction.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

impl PyInstruction {
    pub fn definition(&self) -> Option<CircuitData> {
        Python::attach(|py| -> Option<CircuitData> {
            match self.instruction.getattr(py, intern!(py, "definition")) {
                Ok(definition) => definition
                    .getattr(py, intern!(py, "_data"))
                    .ok()?
                    .extract::<CircuitData>(py)
                    .ok(),
                Err(_) => None,
            }
        })
    }
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: Py<PyAny>,
}

impl PythonOperation for PyGate {
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let deepcopy = imports::DEEPCOPY.get_bound(py);
        Ok(PyGate {
            gate: deepcopy.call1((&self.gate, memo))?.unbind(),
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }

    fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        Ok(PyGate {
            gate: self.gate.call_method0(py, copy_attr)?,
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }
}

impl Operation for PyGate {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn directive(&self) -> bool {
        false
    }
}

impl PyGate {
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        Python::attach(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<Py<PyAny>> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }

    pub fn definition(&self) -> Option<CircuitData> {
        Python::attach(|py| -> Option<CircuitData> {
            match self.gate.getattr(py, intern!(py, "definition")) {
                Ok(definition) => definition
                    .getattr(py, intern!(py, "_data"))
                    .ok()?
                    .extract::<CircuitData>(py)
                    .ok(),
                Err(_) => None,
            }
        })
    }

    pub fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        if self.num_qubits() != 1 {
            return None;
        }
        Python::attach(|py| -> Option<[[Complex64; 2]; 2]> {
            let array = self
                .gate
                .call_method0(py, intern!(py, "to_matrix"))
                .ok()?
                .extract::<PyReadonlyArray2<Complex64>>(py)
                .ok()?;
            let arr = array.as_array();
            Some([[arr[[0, 0]], arr[[0, 1]]], [arr[[1, 0]], arr[[1, 1]]]])
        })
    }
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyOperation {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub operation: Py<PyAny>,
}

impl PythonOperation for PyOperation {
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let deepcopy = imports::DEEPCOPY.get_bound(py);
        Ok(PyOperation {
            operation: deepcopy.call1((&self.operation, memo))?.unbind(),
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }

    fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        Ok(PyOperation {
            operation: self.operation.call_method0(py, copy_attr)?,
            qubits: self.qubits,
            clbits: self.clbits,
            params: self.params,
            op_name: self.op_name.clone(),
        })
    }
}

impl Operation for PyOperation {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn directive(&self) -> bool {
        Python::attach(|py| -> bool {
            match self.operation.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub enum ArrayType {
    NDArray(Array2<Complex64>),
    OneQ(Matrix2<Complex64>),
    TwoQ(Matrix4<Complex64>),
}

/// This class is a rust representation of a UnitaryGate in Python,
/// a gate represented solely by it's unitary matrix.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct UnitaryGate {
    pub array: ArrayType,
}

impl PartialEq for UnitaryGate {
    fn eq(&self, other: &Self) -> bool {
        match (&self.array, &other.array) {
            (ArrayType::OneQ(mat1), ArrayType::OneQ(mat2)) => mat1 == mat2,
            (ArrayType::TwoQ(mat1), ArrayType::TwoQ(mat2)) => mat1 == mat2,
            // we could also slightly optimize comparisons between NDArray and OneQ/TwoQ if
            // this becomes performance critical
            _ => self.matrix() == other.matrix(),
        }
    }
}

impl Operation for UnitaryGate {
    fn name(&self) -> &str {
        "unitary"
    }
    fn num_qubits(&self) -> u32 {
        match &self.array {
            ArrayType::NDArray(arr) => arr.shape()[0].ilog2(),
            ArrayType::OneQ(_) => 1,
            ArrayType::TwoQ(_) => 2,
        }
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        0
    }
    fn directive(&self) -> bool {
        false
    }
}

impl UnitaryGate {
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        match &self.array {
            ArrayType::NDArray(arr) => Some(arr.clone()),
            ArrayType::OneQ(mat) => Some(array!(
                [mat[(0, 0)], mat[(0, 1)]],
                [mat[(1, 0)], mat[(1, 1)]],
            )),
            ArrayType::TwoQ(mat) => Some(array!(
                [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
                [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
                [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
                [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
            )),
        }
    }

    pub fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        match &self.array {
            ArrayType::OneQ(mat) => Some([[mat[(0, 0)], mat[(0, 1)]], [mat[(1, 0)], mat[(1, 1)]]]),
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 1 {
                    Some([[arr[(0, 0)], arr[(0, 1)]], [arr[(1, 0)], arr[(1, 1)]]])
                } else {
                    None
                }
            }
            ArrayType::TwoQ(_) => None,
        }
    }

    pub fn matrix_as_nalgebra_1q(&self) -> Option<Matrix2<Complex64>> {
        match &self.array {
            ArrayType::OneQ(mat) => Some(*mat),
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 1 {
                    Some(Matrix2::new(
                        arr[[0, 0]],
                        arr[[0, 1]],
                        arr[[1, 0]],
                        arr[[1, 1]],
                    ))
                } else {
                    None
                }
            }
            ArrayType::TwoQ(_) => None,
        }
    }
}

impl UnitaryGate {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        if let Some(label) = label {
            kwargs.set_item(intern!(py, "label"), label.into_py_any(py)?)?;
        }
        let out_array = match &self.array {
            ArrayType::NDArray(arr) => arr.to_pyarray(py),
            ArrayType::OneQ(arr) => arr.to_pyarray(py),
            ArrayType::TwoQ(arr) => arr.to_pyarray(py),
        };
        kwargs.set_item(intern!(py, "check_input"), false)?;
        kwargs.set_item(intern!(py, "num_qubits"), self.num_qubits())?;
        let gate = imports::UNITARY_GATE
            .get_bound(py)
            .call((out_array,), Some(&kwargs))?;
        Ok(gate.unbind())
    }

    /// Get a read-only ndarray view of the matrix stored in the `UnitaryGate`
    ///
    /// Regardless of the underlying array type `Matrix2`, `Matrix4`, or `Array2` it returns
    /// a read-only an ndarray `ArrayView2` view to the underlying matrix by reference.
    #[inline]
    pub fn matrix_view(&self) -> ArrayView2<'_, Complex64> {
        match &self.array {
            ArrayType::NDArray(arr) => arr.view(),
            ArrayType::OneQ(mat) => {
                let dim = Dim(mat.shape());
                let strides = Dim(mat.strides());
                // SAFETY: We know the array is a 2x2 and contiguous block so we don't need to
                // check for invalid format
                unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
            }
            ArrayType::TwoQ(mat) => {
                let dim = Dim(mat.shape());
                let strides = Dim(mat.strides());
                // SAFETY: We know the array is a 4x4 and contiguous block so we don't need to
                // check for invalid format
                unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
            }
        }
    }
}

/// This class represents a PauliProductMeasurement instruction.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct PauliProductMeasurement {
    /// The z-component of the pauli.
    pub z: Vec<bool>,
    /// The x-component of the pauli.
    pub x: Vec<bool>,
    /// For a PauliProductMeasurement instruction, the phase of the Pauli can be either 0 or 2,
    /// where the value of 2 corresponds to a sign of `-1`.
    pub neg: bool,
}

impl Operation for PauliProductMeasurement {
    fn name(&self) -> &str {
        "pauli_product_measurement"
    }
    fn num_qubits(&self) -> u32 {
        self.z.len() as u32
    }
    fn num_clbits(&self) -> u32 {
        1
    }
    fn num_params(&self) -> u32 {
        0
    }
    fn directive(&self) -> bool {
        false
    }
}

impl PauliProductMeasurement {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let z = self.z.to_pyarray(py);
        let x = self.x.to_pyarray(py);

        let phase = if self.neg { 2 } else { 0 };

        let py_label = if let Some(label) = label {
            label.into_py_any(py)?
        } else {
            py.None()
        };

        let gate = imports::PAULI_PRODUCT_MEASUREMENT
            .get_bound(py)
            .call_method1(intern!(py, "_from_pauli_data"), (z, x, phase, py_label))?;
        Ok(gate.unbind())
    }
}

impl PartialEq for PauliProductMeasurement {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.z == other.z && self.neg == other.neg
    }
}

impl Eq for PauliProductMeasurement {}
