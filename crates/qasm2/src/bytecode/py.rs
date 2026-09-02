// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
pyo3::import_exception!(qiskit.qasm2.exceptions, QASM2ParseError);

use crate::expr::Expr;
use crate::ext::{ClassicalCallableExt, ClassicalEvaluator};

use crate::parse::ParamId;
use crate::{CustomClassical, CustomInstruction, lex, parse};

use super::InternalBytecode;

/// The Rust parser produces an iterator of these `Bytecode` instructions, which comprise an opcode
/// integer for operation distinction, and a free-form tuple containing the operands.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Bytecode {
    #[pyo3(get)]
    pub opcode: OpCode,
    #[pyo3(get)]
    pub operands: Py<PyAny>,
}

/// The operations that are represented by the "bytecode" passed to Python.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, Eq, PartialEq)]
pub enum OpCode {
    // There is only a `Gate` here, not a `GateInBasis`, because in Python space we don't have the
    // same strict typing requirements to satisfy.
    Gate,
    ConditionedGate,
    Measure,
    ConditionedMeasure,
    Reset,
    ConditionedReset,
    Barrier,
    DeclareQreg,
    DeclareCreg,
    DeclareGate,
    EndDeclareGate,
    DeclareOpaque,
    SpecialInclude,
}

// The following structs, with `Expr` or `OpCode` in the name (but not the top-level `OpCode`
// above) build up the tree of symbolic expressions for the parameter applications within gate
// bodies.  We choose to store this in the gate classes that the Python component emits, so it can
// lazily create definitions as required, rather than eagerly binding them as the file is parsed.
//
// In Python space we would usually have the classes inherit from some shared subclass, but doing
// that makes things a little fiddlier with PyO3, and there's no real benefit for our uses.

/// A (potentially folded) floating-point constant value as part of an expression.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprConstant {
    #[pyo3(get)]
    pub value: f64,
}

/// A reference to one of the arguments to the gate.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprArgument {
    #[pyo3(get)]
    pub index: ParamId,
}

/// A unary operation acting on some other part of the expression tree.  This includes the `+` and
/// `-` unary operators, but also any of the built-in scientific-calculator functions.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprUnary {
    #[pyo3(get)]
    pub opcode: UnaryOpCode,
    #[pyo3(get)]
    pub argument: Py<PyAny>,
}

/// A binary operation acting on two other parts of the expression tree.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprBinary {
    #[pyo3(get)]
    pub opcode: BinaryOpCode,
    #[pyo3(get)]
    pub left: Py<PyAny>,
    #[pyo3(get)]
    pub right: Py<PyAny>,
}

/// Some custom callable Python function that the user told us about.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprCustom {
    pub callable: ClassicalCallableExt,
    #[pyo3(get)]
    pub arguments: Vec<Py<PyAny>>,
}

#[pymethods]
impl ExprCustom {
    /// Invoke the custom callable with pre-evaluated float arguments.
    fn call(&self, py: Python<'_>, args: Vec<f64>) -> PyResult<f64> {
        Ok(ClassicalEvaluator::attached(py).eval(&self.callable, &args)?)
    }
}

/// Discriminator for the different types of unary operator.  We could have a separate class for
/// each of these, but this way involves fewer imports in Python, and also serves to split up the
/// option tree at the top level, so we don't have to test every unary operator before testing
/// other operations.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum UnaryOpCode {
    Negate,
    Cos,
    Exp,
    Ln,
    Sin,
    Sqrt,
    Tan,
}

/// Discriminator for the different types of binary operator.  We could have a separate class for
/// each of these, but this way involves fewer imports in Python, and also serves to split up the
/// option tree at the top level, so we don't have to test every binary operator before testing
/// other operations.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum BinaryOpCode {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

/// The custom iterator object that is returned up to Python space for iteration through the
/// bytecode stream.  This is never constructed on the Python side; it is built in Rust space
/// by Python calls to [bytecode_from_string] and [bytecode_from_file].
#[pyclass]
pub struct BytecodeIterator {
    parser_state: parse::State,
    buffer: Vec<Option<InternalBytecode>>,
    buffer_used: usize,
}

impl BytecodeIterator {
    pub fn new(
        tokens: lex::TokenStream,
        include_path: Vec<std::path::PathBuf>,
        custom_instructions: &[CustomInstruction],
        custom_classical: &[CustomClassical],
        strict: bool,
    ) -> PyResult<Self> {
        Ok(BytecodeIterator {
            parser_state: parse::State::new(
                tokens,
                include_path,
                custom_instructions,
                custom_classical,
                strict,
            )?,
            buffer: vec![],
            buffer_used: 0,
        })
    }
}

#[pymethods]
impl BytecodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Bytecode>> {
        if self.buffer_used >= self.buffer.len() {
            self.buffer.clear();
            self.buffer_used = 0;
            self.parser_state
                .parse_next(&mut self.buffer, ClassicalEvaluator::attached(py))?;
        }
        if self.buffer.is_empty() {
            Ok(None)
        } else {
            self.buffer_used += 1;
            Ok(self.buffer[self.buffer_used - 1]
                .take()
                .map(|bytecode| bytecode.into_pyobject(py))
                .transpose()?
                .map(|x| x.get().clone()))
        }
    }
}

#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
pub struct GateBodyArguments(Vec<Expr>);

impl GateBodyArguments {
    pub(super) fn new(arguments: Vec<Expr>) -> Self {
        Self(arguments)
    }
}

#[pymethods]
impl GateBodyArguments {
    fn evaluate(&self, params: Vec<f64>, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.0
            .iter()
            .map(|expr| {
                crate::expr::evaluate(expr, &params, ClassicalEvaluator::attached(py))
                    .map_err(PyErr::from)
            })
            .collect()
    }
}
