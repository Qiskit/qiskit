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

use crate::circuit_data::CircuitData;
use crate::imports::get_std_gate_class;
use crate::imports::{PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
use crate::{gate_matrix, impl_intopyobject_for_copy_pyclass, Qubit};
use approx::relative_eq;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::Deref;
use std::{fmt, vec};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{array, aview2, Array2};
use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

use crate::bit::{ClassicalRegister, ShareableClbit};
use crate::classical::expr;
use crate::duration::Duration;
use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyFloat, PyIterator, PyList, PyTuple};
use pyo3::{intern, Python};

#[derive(Clone, Debug, IntoPyObject)]
pub enum Param {
    ParameterExpression(PyObject),
    Float(f64),
    Circuit(PyObject),
    Condition(Condition),
    Duration(Duration),
    Target(Target),
    Indexset(Vec<usize>),
    Obj(PyObject),
}

// impl<'a, 'py> IntoPyObject<'py> for &'a Param {
//     type Target = PyAny;
//     type Output = Borrowed<'a, 'py, Self::Target>;
//     type Error = PyErr;
//
//     fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
//         match self {
//             Param::ParameterExpression(e) => Ok(e.bind_borrowed(py)),
//             Param::Float(f) => f.into_bound_py_any(py),
//             Param::Circuit(c) => c.un
//             Param::Condition(_) => {}
//             Param::Duration(_) => {}
//             Param::Target(_) => {}
//             Param::Obj(_) => {}
//         }
//     }
// }

impl Param {
    pub fn eq(&self, py: Python, other: &Param) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(a == b),
            [Self::Float(a), Self::ParameterExpression(b)] => b.bind(py).eq(a),
            [Self::ParameterExpression(a), Self::Float(b)] => a.bind(py).eq(b),
            [Self::ParameterExpression(a), Self::ParameterExpression(b)] => a.bind(py).eq(b),
            [Self::Obj(_), Self::Float(_)] => Ok(false),
            [Self::Float(_), Self::Obj(_)] => Ok(false),
            [Self::Obj(a), Self::ParameterExpression(b)] => a.bind(py).eq(b),
            [Self::Obj(a), Self::Obj(b)] => a.bind(py).eq(b),
            [Self::ParameterExpression(a), Self::Obj(b)] => a.bind(py).eq(b),
            [Self::Condition(a), Self::Condition(b)] => Ok(a == b),
            [Self::Target(a), Self::Target(b)] => Ok(a == b),
            [Self::Duration(a), Self::Duration(b)] => Ok(a == b),
            [Self::Circuit(a), Self::Circuit(b)] => a.bind(py).eq(b),
            [Self::Indexset(a), Self::Indexset(b)] => Ok(a == b),
            _ => todo!(),
        }
    }

    pub fn is_close(&self, py: Python, other: &Param, max_relative: f64) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(relative_eq!(a, b, max_relative = max_relative)),
            _ => self.eq(py, other),
        }
    }
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))? {
            Param::ParameterExpression(b.clone().unbind())
        } else if let Ok(val) = b.extract::<f64>() {
            Param::Float(val)
        } else {
            // TODO: write the rest of the extraction logic needed
            Param::Obj(b.clone().unbind())
        })
    }
}

impl Param {
    /// Get an iterator over any Python-space `Parameter` instances tracked within this `Param`.
    pub fn iter_parameters<'py>(&self, py: Python<'py>) -> PyResult<ParamParameterIter<'py>> {
        let parameters_attr = intern!(py, "parameters");
        match self {
            Param::Float(_) => Ok(ParamParameterIter(None)),
            Param::ParameterExpression(expr) => Ok(ParamParameterIter(Some(
                expr.bind(py).getattr(parameters_attr)?.try_iter()?,
            ))),
            Param::Obj(obj) => {
                let obj = obj.bind(py);
                if obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                    Ok(ParamParameterIter(Some(
                        obj.getattr(parameters_attr)?.try_iter()?,
                    )))
                } else {
                    Ok(ParamParameterIter(None))
                }
            }
            Param::Circuit(circuit) => todo!(),
            _ => Ok(ParamParameterIter(None)),
        }
    }

    /// Extract from a Python object without numeric coercion to float.  The default conversion will
    /// coerce integers into floats, but in things like `assign_parameters`, this is not always
    /// desirable.
    pub fn extract_no_coerce(ob: &Bound<PyAny>) -> PyResult<Self> {
        Ok(if ob.is_instance_of::<PyFloat>() {
            Param::Float(ob.extract()?)
        } else if ob.is_instance(PARAMETER_EXPRESSION.get_bound(ob.py()))? {
            Param::ParameterExpression(ob.clone().unbind())
        } else {
            Param::Obj(ob.clone().unbind())
        })
    }

    /// Clones the [Param] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone_ref(py)),
            Param::Float(float) => Param::Float(*float),
            Param::Obj(obj) => Param::Obj(obj.clone_ref(py)),
            _ => todo!(),
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

/// Struct to provide iteration over Python-space `Parameter` instances within a `Param`.
pub struct ParamParameterIter<'py>(Option<Bound<'py, PyIterator>>);
impl<'py> Iterator for ParamParameterIter<'py> {
    type Item = PyResult<Bound<'py, PyAny>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(|iter| iter.next())
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn standard_gate(&self) -> Option<StandardGate>;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
#[derive(Debug)]
pub enum OperationRef<'a> {
    ControlFlow(&'a ControlFlow),
    StandardGate(StandardGate),
    StandardInstruction(StandardInstruction),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(&'a UnitaryGate),
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
        }
    }
    #[inline]
    fn standard_gate(&self) -> Option<StandardGate> {
        match self {
            Self::ControlFlow(op) => op.standard_gate(),
            Self::StandardGate(standard) => standard.standard_gate(),
            Self::StandardInstruction(instruction) => instruction.standard_gate(),
            Self::Gate(gate) => gate.standard_gate(),
            Self::Instruction(instruction) => instruction.standard_gate(),
            Self::Operation(operation) => operation.standard_gate(),
            Self::Unitary(unitary) => unitary.standard_gate(),
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
        }
    }
}

// TODO: make this Copy once ControlFlowRef doesn't need Vec
#[derive(Debug)]
pub enum InstructionRef<'a, T> {
    ControlFlow(ControlFlowRef<'a, T>),
    StandardGate(StandardGateRef<'a>),
    StandardInstruction(StandardInstructionRef<'a>),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(UnitaryGateRef<'a>),
}

impl<'a, T> InstructionRef<'a, T> {
    pub fn standard_gate(self) -> Option<StandardGateRef<'a>> {
        match self {
            InstructionRef::StandardGate(standard) => Some(standard),
            _ => None,
        }
    }

    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        match self {
            InstructionRef::StandardGate(g) => g.matrix(),
            InstructionRef::Gate(g) => g.matrix(),
            InstructionRef::Unitary(u) => u.matrix(),
            _ => None,
        }
    }

    pub fn definition(&self) -> Option<CircuitData> {
        match self {
            InstructionRef::StandardGate(g) => g.definition(),
            InstructionRef::Gate(g) => g.definition(),
            InstructionRef::Instruction(i) => i.definition(),
            _ => None,
        }
    }

    // TODO:
    /// Does this instruction contain any compile-time symbolic `ParameterExpression`s?
    pub fn is_parameterized(&self) -> bool {
        todo!()
        // match self {
        //
        // }
        // self.params_view()
        //     .iter()
        //     .any(|x| matches!(x, Param::ParameterExpression(_)))
    }
}

impl<T: PyEq> PyEq for InstructionRef<'_, T> {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        match (self, other) {
            (InstructionRef::ControlFlow(c1), InstructionRef::ControlFlow(c2)) => c1.py_eq(py, c2),
            (InstructionRef::StandardGate(g1), InstructionRef::StandardGate(g2)) => {
                g1.py_eq(py, g2)
            }
            (InstructionRef::StandardInstruction(i1), InstructionRef::StandardInstruction(i2)) => {
                i1.py_eq(py, i2)
            }
            (InstructionRef::Gate(g1), InstructionRef::Gate(g2)) => g1.py_eq(py, g2),
            (InstructionRef::Instruction(i1), InstructionRef::Instruction(i2)) => i1.py_eq(py, i2),
            (InstructionRef::Operation(o1), InstructionRef::Operation(o2)) => o1.py_eq(py, o2),
            (InstructionRef::Unitary(u1), InstructionRef::Unitary(u2)) => u1.py_eq(py, u2),
            _ => Ok(false),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[repr(align(8))]
pub enum ControlFlow {
    Box {
        duration: Duration,
        qubits: u32,
        clbits: u32,
    },
    BreakLoop {
        qubits: u32,
        clbits: u32,
    },
    ContinueLoop {
        qubits: u32,
        clbits: u32,
    },
    ForLoop {
        qubits: u32,
        clbits: u32,
    },
    IfElse {
        condition: Condition,
        qubits: u32,
        clbits: u32,
    },
    Switch {
        target: Target,
        qubits: u32,
        clbits: u32,
        cases: u32,
    },
    While {
        condition: Condition,
        qubits: u32,
        clbits: u32,
    },
}

impl Operation for ControlFlow {
    fn name(&self) -> &str {
        match self {
            ControlFlow::Box { .. } => "box",
            ControlFlow::BreakLoop { .. } => "break_loop",
            ControlFlow::ContinueLoop { .. } => "continue_loop",
            ControlFlow::ForLoop { .. } => "for_loop",
            ControlFlow::IfElse { .. } => "if_else",
            ControlFlow::Switch { .. } => "switch_case",
            ControlFlow::While { .. } => "while_loop",
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            ControlFlow::Box { qubits, .. }
            | ControlFlow::BreakLoop { qubits, .. }
            | ControlFlow::ContinueLoop { qubits, .. }
            | ControlFlow::ForLoop { qubits, .. }
            | ControlFlow::IfElse { qubits, .. }
            | ControlFlow::Switch { qubits, .. }
            | ControlFlow::While { qubits, .. } => *qubits,
        }
    }

    fn num_clbits(&self) -> u32 {
        match self {
            ControlFlow::Box { clbits, .. }
            | ControlFlow::BreakLoop { clbits, .. }
            | ControlFlow::ContinueLoop { clbits, .. }
            | ControlFlow::ForLoop { clbits, .. }
            | ControlFlow::IfElse { clbits, .. }
            | ControlFlow::Switch { clbits, .. }
            | ControlFlow::While { clbits, .. } => *clbits,
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            ControlFlow::Box { .. } => 1,
            ControlFlow::BreakLoop { .. } => 0,
            ControlFlow::ContinueLoop { .. } => 0,
            ControlFlow::ForLoop { .. } => 3,
            ControlFlow::IfElse { .. } => 2,
            ControlFlow::Switch { cases, .. } => *cases,
            ControlFlow::While { .. } => 1,
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        false
    }
}

/// A control flow operation's condition.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum Condition {
    Bit(ShareableClbit, usize),
    Register(ClassicalRegister, usize),
    Expr(expr::Expr),
}

impl<'py> FromPyObject<'py> for Condition {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok((bit, value)) = ob.extract::<(ShareableClbit, usize)>() {
            Ok(Condition::Bit(bit, value))
        } else if let Ok((register, value)) = ob.extract::<(ClassicalRegister, usize)>() {
            Ok(Condition::Register(register, value))
        } else {
            Ok(Condition::Expr(ob.extract()?))
        }
    }
}

/// A control flow operation's target.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum Target {
    Bit(ShareableClbit),
    Register(ClassicalRegister),
    Expr(expr::Expr),
}

impl<'py> FromPyObject<'py> for Target {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(bit) = ob.extract::<ShareableClbit>() {
            Ok(Target::Bit(bit))
        } else if let Ok(register) = ob.extract::<ClassicalRegister>() {
            Ok(Target::Register(register))
        } else {
            Ok(Target::Expr(ob.extract()?))
        }
    }
}

#[derive(Clone, Debug)]
pub enum ControlFlowRef<'a, T> {
    Box(&'a Duration, &'a T),
    BreakLoop,
    ContinueLoop,
    ForLoop {
        indexset: &'a [usize],
        loop_param: &'a PyObject,
        body: &'a T,
    },
    IfElse {
        condition: &'a Condition,
        true_body: &'a T,
        false_body: &'a T,
    },
    Switch {
        target: &'a Target,
        cases: Vec<&'a T>,
    },
    While {
        condition: &'a Condition,
        body: &'a T,
    },
}

pub trait PyEq {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool>;
}

impl PyEq for Py<PyAny> {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.bind(py).eq(other)
    }
}

impl<'a, T> ControlFlowRef<'a, T> {
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = &'a T> {
        match self {
            ControlFlowRef::Box(_, body) => vec![*body],
            ControlFlowRef::BreakLoop => vec![],
            ControlFlowRef::ContinueLoop => vec![],
            ControlFlowRef::ForLoop { body, .. } => vec![*body],
            ControlFlowRef::IfElse {
                true_body,
                false_body,
                ..
            } => vec![*true_body, *false_body],
            ControlFlowRef::Switch { cases, .. } => cases.clone(),
            ControlFlowRef::While { body, .. } => vec![*body],
        }
        .into_iter()
    }
}

impl<T: PyEq> PyEq for ControlFlowRef<'_, T> {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        match (self, other) {
            (ControlFlowRef::Box(duration_a, body_a), ControlFlowRef::Box(duration_b, body_b)) => {
                Ok(duration_a == duration_b && body_a.py_eq(py, body_b)?)
            }
            (ControlFlowRef::BreakLoop, ControlFlowRef::BreakLoop) => Ok(true),
            (ControlFlowRef::ContinueLoop, ControlFlowRef::ContinueLoop) => Ok(true),
            (
                ControlFlowRef::ForLoop {
                    indexset: indexset_a,
                    loop_param: loop_param_a,
                    body: body_a,
                },
                ControlFlowRef::ForLoop {
                    indexset: indexset_b,
                    loop_param: loop_param_b,
                    body: body_b,
                },
            ) => Ok(indexset_a == indexset_b
                && loop_param_a.bind(py).eq(loop_param_b)?
                && body_a.py_eq(py, body_b)?),
            (
                ControlFlowRef::IfElse {
                    condition: condition_a,
                    true_body: true_body_a,
                    false_body: false_body_a,
                },
                ControlFlowRef::IfElse {
                    condition: condition_b,
                    true_body: true_body_b,
                    false_body: false_body_b,
                },
            ) => Ok(condition_a == condition_b
                && true_body_a.py_eq(py, true_body_b)?
                && false_body_a.py_eq(py, false_body_b)?),
            (
                ControlFlowRef::Switch {
                    target: target_a,
                    cases: cases_a,
                },
                ControlFlowRef::Switch {
                    target: target_b,
                    cases: cases_b,
                },
            ) => {
                if target_a != target_b || cases_a.len() != cases_b.len() {
                    return Ok(false);
                }
                for (a, b) in cases_a.iter().zip(cases_b.iter()) {
                    if !a.py_eq(py, b)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (
                ControlFlowRef::While {
                    condition: condition_a,
                    body: body_a,
                },
                ControlFlowRef::While {
                    condition: condition_b,
                    body: body_b,
                },
            ) => Ok(condition_a == condition_b && body_a.py_eq(py, body_b)?),
            _ => Ok(false),
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

impl<'py> FromPyObject<'py> for DelayUnit {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
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
                    "Unit '{}' is invalid.",
                    unknown_unit
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

    fn standard_gate(&self) -> Option<StandardGate> {
        None
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

#[derive(Debug)]
pub enum StandardInstructionRef<'a> {
    Barrier(u32),
    Delay {
        duration: &'a Param,
        unit: DelayUnit,
    },
    Measure,
    Reset,
}

impl<'a> PyEq for StandardInstructionRef<'a> {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        match (self, other) {
            (StandardInstructionRef::Barrier(a), StandardInstructionRef::Barrier(b)) => Ok(a == b),
            (
                StandardInstructionRef::Delay {
                    duration: duration_a,
                    unit: unit_a,
                },
                StandardInstructionRef::Delay {
                    duration: duration_b,
                    unit: unit_b,
                },
            ) => Ok(unit_a == unit_b && duration_a.is_close(py, duration_b, 1e-10)?),
            (StandardInstructionRef::Measure, StandardInstructionRef::Measure) => Ok(true),
            (StandardInstructionRef::Reset, StandardInstructionRef::Reset) => Ok(true),
            _ => Ok(false),
        }
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
        params: Option<&[Param]>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = get_std_gate_class(py, *self)?;
        let args = match params.unwrap_or(&[]) {
            &[] => PyTuple::empty(py),
            params => PyTuple::new(
                py,
                params.iter().map(|x| x.clone().into_pyobject(py).unwrap()),
            )?,
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
                [Param::Float(theta), Param::Float(phi), Param::Float(lam), Param::Float(gamma)] => {
                    Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned())
                }
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
            Self::GlobalPhase => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(py, 0, [], params[0].clone())
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::H => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::U,
                            smallvec![Param::Float(PI / 2.), FLOAT_ZERO, Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::I => None,
            Self::X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::U,
                            smallvec![Param::Float(PI), FLOAT_ZERO, Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::Y => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                )
            }),

            Self::Z => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            smallvec![Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::Phase => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::U,
                            smallvec![FLOAT_ZERO, FLOAT_ZERO, params[0].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::R => Python::with_gil(|py| -> Option<CircuitData> {
                let theta_expr = clone_param(&params[0], py);
                let phi_expr1 = add_param(&params[1], -PI / 2., py);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0, py);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(Self::U, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RX => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::RY => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::RZ => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(Self::Phase, smallvec![theta.clone()], smallvec![Qubit(0)])],
                        multiply_param(theta, -0.5, py),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::S => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            smallvec![Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::Sdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            smallvec![Param::Float(-PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                            (Self::H, smallvec![], smallvec![Qubit(0)]),
                            (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::S, smallvec![], smallvec![Qubit(0)]),
                            (Self::H, smallvec![], smallvec![Qubit(0)]),
                            (Self::S, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::T => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            smallvec![Param::Float(PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::Tdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            smallvec![Param::Float(-PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U => None,
            Self::U1 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::Phase,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U2 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::U,
                            smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U3 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::U,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CH => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),

            Self::CX => None,
            Self::CY => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::CZ => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::DCX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ECR => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZX,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                            (Self::X, smallvec![], smallvec![Qubit(0)]),
                            (
                                Self::RZX,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::Swap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ISwap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                )
            }),
            Self::CPhase => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::Phase,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q0,
                            ),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (
                                Self::Phase,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CX, smallvec![], q0_1),
                            (
                                Self::Phase,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CRX => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::Phase,
                                smallvec![Param::Float(PI / 2.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(theta, -0.5, py),
                                    Param::Float(0.0),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(theta, 0.5, py),
                                    Param::Float(-PI / 2.),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRY => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRZ => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CS => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::Phase, smallvec![Param::Float(PI / 4.)], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::Phase, smallvec![Param::Float(-PI / 4.)], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Phase, smallvec![Param::Float(PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSdg => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::Phase, smallvec![Param::Float(-PI / 4.)], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::Phase, smallvec![Param::Float(PI / 4.)], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Phase, smallvec![Param::Float(-PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSX => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CPhase, smallvec![Param::Float(PI / 2.)], q0_1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU => Python::with_gil(|py| -> Option<CircuitData> {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
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
            }),
            Self::CU1 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::U1,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(0)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU3 => Python::with_gil(|py| -> Option<CircuitData> {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::U1, smallvec![param_first_u1], smallvec![Qubit(0)]),
                            (Self::U1, smallvec![param_second_u1], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
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
            }),
            Self::RXX => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::RYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RX, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::RX, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::RX, smallvec![Param::Float(-PI / 2.)], q0),
                            (Self::RX, smallvec![Param::Float(-PI / 2.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZZ => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::RZX => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),
            Self::XXMinusYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZ,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q1.clone(),
                            ),
                            (Self::RZ, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SX, smallvec![], q0.clone()),
                            (Self::RZ, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::S, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5, py)],
                                q0.clone(),
                            ),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::RZ, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXdg, smallvec![], q0.clone()),
                            (Self::RZ, smallvec![Param::Float(PI / 2.)], q0),
                            (Self::RZ, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXPlusYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RZ, smallvec![beta.clone()], q0.clone()),
                            (Self::RZ, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SX, smallvec![], q1.clone()),
                            (Self::RZ, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::S, smallvec![], q0.clone()),
                            (Self::CX, smallvec![], q1_0.clone()),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q0.clone(),
                            ),
                            (Self::CX, smallvec![], q1_0),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::RZ, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXdg, smallvec![], q1.clone()),
                            (Self::RZ, smallvec![Param::Float(PI / 2.)], q1),
                            (Self::RZ, smallvec![multiply_param(beta, -1.0, py)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CCX => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
            }),

            Self::CCZ => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                )
            }),
            Self::CSwap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                )
            }),

            Self::RCCX => Python::with_gil(|py| -> Option<CircuitData> {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (
                                Self::U2,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                q2.clone(),
                            ),
                            (Self::U1, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CX, smallvec![], q1_2.clone()),
                            (Self::U1, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (Self::CX, smallvec![], q0_2),
                            (Self::U1, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CX, smallvec![], q1_2),
                            (Self::U1, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (Self::U2, smallvec![FLOAT_ZERO, Param::Float(PI)], q2),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::C3X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
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
                )
            }),

            Self::C3SX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(0), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::H, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RC3X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (
                                Self::U2,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
        }
    }

    pub fn inverse(&self, params: &[Param]) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        match self {
            Self::GlobalPhase => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::GlobalPhase,
                    smallvec![multiply_param(&params[0], -1.0, py)],
                )
            })),
            Self::H => Some((Self::H, smallvec![])),
            Self::I => Some((Self::I, smallvec![])),
            Self::X => Some((Self::X, smallvec![])),
            Self::Y => Some((Self::Y, smallvec![])),
            Self::Z => Some((Self::Z, smallvec![])),
            Self::Phase => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::Phase, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::R => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::R,
                    smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                )
            })),
            Self::RX => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RX, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::RY => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RY, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::RZ => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RZ, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::S => Some((Self::Sdg, smallvec![])),
            Self::Sdg => Some((Self::S, smallvec![])),
            Self::SX => Some((Self::SXdg, smallvec![])),
            Self::SXdg => Some((Self::SX, smallvec![])),
            Self::T => Some((Self::Tdg, smallvec![])),
            Self::Tdg => Some((Self::T, smallvec![])),
            Self::U => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::U,
                    smallvec![
                        multiply_param(&params[0], -1.0, py),
                        multiply_param(&params[2], -1.0, py),
                        multiply_param(&params[1], -1.0, py),
                    ],
                )
            })),
            Self::U1 => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::U1, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::U2 => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::U2,
                    smallvec![
                        add_param(&multiply_param(&params[1], -1.0, py), -PI, py),
                        add_param(&multiply_param(&params[0], -1.0, py), PI, py),
                    ],
                )
            })),
            Self::U3 => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::U3,
                    smallvec![
                        multiply_param(&params[0], -1.0, py),
                        multiply_param(&params[2], -1.0, py),
                        multiply_param(&params[1], -1.0, py),
                    ],
                )
            })),
            Self::CH => Some((Self::CH, smallvec![])),
            Self::CX => Some((Self::CX, smallvec![])),
            Self::CY => Some((Self::CY, smallvec![])),
            Self::CZ => Some((Self::CZ, smallvec![])),
            Self::DCX => None, // the inverse in not a StandardGate
            Self::ECR => Some((Self::ECR, smallvec![])),
            Self::Swap => Some((Self::Swap, smallvec![])),
            Self::ISwap => None, // the inverse in not a StandardGate
            Self::CPhase => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::CPhase,
                    smallvec![multiply_param(&params[0], -1.0, py)],
                )
            })),
            Self::CRX => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::CRX, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::CRY => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::CRY, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::CRZ => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::CRZ, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::CS => Some((Self::CSdg, smallvec![])),
            Self::CSdg => Some((Self::CS, smallvec![])),
            Self::CSX => None, // the inverse in not a StandardGate
            Self::CU => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::CU,
                    smallvec![
                        multiply_param(&params[0], -1.0, py),
                        multiply_param(&params[2], -1.0, py),
                        multiply_param(&params[1], -1.0, py),
                        multiply_param(&params[3], -1.0, py),
                    ],
                )
            })),
            Self::CU1 => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::CU1, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::CU3 => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::CU3,
                    smallvec![
                        multiply_param(&params[0], -1.0, py),
                        multiply_param(&params[2], -1.0, py),
                        multiply_param(&params[1], -1.0, py),
                    ],
                )
            })),
            Self::RXX => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RXX, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::RYY => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RYY, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::RZZ => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RZZ, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::RZX => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (Self::RZX, smallvec![multiply_param(&params[0], -1.0, py)])
            })),
            Self::XXMinusYY => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::XXMinusYY,
                    smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                )
            })),
            Self::XXPlusYY => Some(Python::with_gil(|py| -> (Self, SmallVec<[Param; 3]>) {
                (
                    Self::XXPlusYY,
                    smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                )
            })),
            Self::CCX => Some((Self::CCX, smallvec![])),
            Self::CCZ => Some((Self::CCZ, smallvec![])),
            Self::CSwap => Some((Self::CSwap, smallvec![])),
            Self::RCCX => None, // the inverse in not a StandardGate
            Self::C3X => Some((Self::C3X, smallvec![])),
            Self::C3SX => None, // the inverse in not a StandardGate
            Self::RC3X => None, // the inverse in not a StandardGate
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
        get_std_gate_class(py, *self)
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
    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }
    fn directive(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct StandardGateRef<'a>(pub StandardGate, pub &'a [Param]);

impl<'a> StandardGateRef<'a> {
    #[inline]
    pub fn gate(&self) -> &StandardGate {
        &self.0
    }

    #[inline]
    pub fn params(&self) -> &'a [Param] {
        self.1
    }

    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        self.0.matrix(self.1)
    }

    pub fn definition(&self) -> Option<CircuitData> {
        self.0.definition(self.1)
    }
}

impl<'a> PyEq for StandardGateRef<'a> {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        if self.0 != other.0 || self.1.len() != other.1.len() {
            return Ok(false);
        }
        for (a, b) in self.1.iter().zip(other.1.iter()) {
            // TODO: make sure this is right max_rel
            if !a.is_close(py, b, 1e-10)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

// Return explicitly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &Param, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone_ref(py)),
        _ => unreachable!(),
    }
}

/// Multiply a ``Param`` with a float.
pub fn multiply_param(param: &Param, mult: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(theta * mult),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__rmul__"), (mult,))
                .expect("Multiplication of Parameter expression by float failed."),
        ),
        _ => unreachable!("Unsupported multiplication."),
    }
}

/// Multiply two ``Param``s.
pub fn multiply_params(param1: Param, param2: Param, py: Python) -> Param {
    match (&param1, &param2) {
        (Param::Float(theta), Param::Float(lambda)) => Param::Float(theta * lambda),
        (param, Param::Float(theta)) => multiply_param(param, *theta, py),
        (Param::Float(theta), param) => multiply_param(param, *theta, py),
        (Param::ParameterExpression(p1), Param::ParameterExpression(p2)) => {
            Param::ParameterExpression(
                p1.clone_ref(py)
                    .call_method1(py, intern!(py, "__rmul__"), (p2,))
                    .expect("Parameter expression multiplication failed"),
            )
        }
        _ => unreachable!("Unsupported multiplication."),
    }
}

pub fn add_param(param: &Param, summand: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (summand,))
                .expect("Sum of Parameter expression and float failed."),
        ),
        _ => unreachable!(),
    }
}

pub fn radd_param(param1: Param, param2: Param, py: Python) -> Param {
    match [&param1, &param2] {
        [Param::Float(theta), Param::Float(lambda)] => Param::Float(theta + lambda),
        [Param::Float(theta), Param::ParameterExpression(_lambda)] => {
            add_param(&param2, *theta, py)
        }
        [Param::ParameterExpression(_theta), Param::Float(lambda)] => {
            add_param(&param1, *lambda, py)
        }
        [Param::ParameterExpression(theta), Param::ParameterExpression(lambda)] => {
            Param::ParameterExpression(
                theta
                    .clone_ref(py)
                    .call_method1(py, intern!(py, "__radd__"), (lambda,))
                    .expect("Parameter expression addition failed"),
            )
        }
        _ => unreachable!(),
    }
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
    pub instruction: PyObject,
}

impl PyEq for PyInstruction {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.instruction.bind(py).eq(&other.instruction)
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
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }
    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
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
    fn definition(&self) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
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
    pub gate: PyObject,
}

impl PyEq for PyGate {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.gate.bind(py).eq(&other.gate)
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
    fn standard_gate(&self) -> Option<StandardGate> {
        Python::with_gil(|py| -> Option<StandardGate> {
            match self.gate.getattr(py, intern!(py, "_standard_gate")) {
                Ok(stdgate) => stdgate.extract(py).unwrap_or_default(),
                Err(_) => None,
            }
        })
    }
    fn directive(&self) -> bool {
        false
    }
}

impl PyGate {
    fn matrix(&self) -> Option<Array2<Complex64>> {
        Python::with_gil(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<PyObject> = to_matrix.call0(py).ok()?.extract(py).ok();
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

    fn definition(&self) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
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
    pub operation: PyObject,
}

impl PyEq for PyOperation {
    fn py_eq(&self, py: Python, other: &Self) -> PyResult<bool> {
        self.operation.bind(py).eq(&other.operation)
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
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }
    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
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
    fn standard_gate(&self) -> Option<StandardGate> {
        None
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
}

#[derive(Debug)]
pub struct UnitaryGateRef<'a>(pub &'a UnitaryGate);

impl Deref for UnitaryGateRef<'_> {
    type Target = UnitaryGate;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
//
// impl<'a> UnitaryGateRef<'a> {
//     pub fn matrix(&self) -> Option<Array2<Complex64>> {
//         self.0.matrix()
//     }
// }

impl<'a> PyEq for UnitaryGateRef<'a> {
    fn py_eq(&self, _py: Python, other: &Self) -> PyResult<bool> {
        match [&self.array, &other.array] {
            [ArrayType::NDArray(a), ArrayType::NDArray(b)] => {
                Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
            }
            [ArrayType::OneQ(a), ArrayType::NDArray(b)]
            | [ArrayType::NDArray(b), ArrayType::OneQ(a)] => {
                if b.shape()[0] == 2 {
                    for i in 0..2 {
                        for j in 0..2 {
                            if !relative_eq!(
                                b[[i, j]],
                                a[(i, j)],
                                max_relative = 1e-5,
                                epsilon = 1e-8
                            ) {
                                return Ok(false);
                            }
                        }
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            [ArrayType::TwoQ(a), ArrayType::NDArray(b)]
            | [ArrayType::NDArray(b), ArrayType::TwoQ(a)] => {
                if b.shape()[0] == 4 {
                    for i in 0..4 {
                        for j in 0..4 {
                            if !relative_eq!(
                                b[[i, j]],
                                a[(i, j)],
                                max_relative = 1e-5,
                                epsilon = 1e-8
                            ) {
                                return Ok(false);
                            }
                        }
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            [ArrayType::OneQ(a), ArrayType::OneQ(b)] => {
                Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
            }
            [ArrayType::TwoQ(a), ArrayType::TwoQ(b)] => {
                Ok(relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8))
            }
            _ => Ok(false),
        }
    }
}
