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

use crate::error::ParseError;
use crate::expr::Expr;
use crate::{CustomClassical, CustomInstruction, lex, parse};

use super::InternalBytecode;

/// Convert a `ParseError` from the pyo3-free parsing modules into the `QASM2ParseError`
/// Python exception, at the boundary where results cross back into Python space.
impl From<ParseError> for PyErr {
    fn from(e: ParseError) -> PyErr {
        let py_err = QASM2ParseError::new_err(e.message);
        if let Some(source) = e.source {
            Python::attach(|py| py_err.set_cause(py, Some(*source)));
        }
        py_err
    }
}

/// The Rust parser produces an iterator of these `Bytecode` instructions, which comprise an opcode
/// integer for operation distinction, and a free-form tuple containing the operands.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Bytecode {
    #[pyo3(get)]
    opcode: OpCode,
    #[pyo3(get)]
    operands: Py<PyAny>,
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

#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
pub struct GateBodyArguments(Vec<Expr>);

#[pymethods]
impl GateBodyArguments {
    fn evaluate(&self, params: Vec<f64>, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.0
            .iter()
            .map(|expr| crate::expr::evaluate(expr, &params, py).map_err(PyErr::from))
            .collect()
    }
}

impl<'py> IntoPyObject<'py> for InternalBytecode {
    type Target = Bytecode;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    /// Convert the internal bytecode representation to a Python-space one.
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(
            py,
            match self {
                InternalBytecode::Gate {
                    id,
                    arguments,
                    qubits,
                } => Bytecode {
                    opcode: OpCode::Gate,
                    operands: (id, arguments, qubits)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::ConditionedGate {
                    id,
                    arguments,
                    qubits,
                    creg,
                    value,
                } => Bytecode {
                    opcode: OpCode::ConditionedGate,
                    operands: (id, arguments, qubits, creg, value)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::Measure { qubit, clbit } => Bytecode {
                    opcode: OpCode::Measure,
                    operands: (qubit, clbit).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::ConditionedMeasure {
                    qubit,
                    clbit,
                    creg,
                    value,
                } => Bytecode {
                    opcode: OpCode::ConditionedMeasure,
                    operands: (qubit, clbit, creg, value)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::Reset { qubit } => Bytecode {
                    opcode: OpCode::Reset,
                    operands: (qubit,).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::ConditionedReset { qubit, creg, value } => Bytecode {
                    opcode: OpCode::ConditionedReset,
                    operands: (qubit, creg, value).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::Barrier { qubits } => Bytecode {
                    opcode: OpCode::Barrier,
                    operands: (qubits,).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareQreg { name, size } => Bytecode {
                    opcode: OpCode::DeclareQreg,
                    operands: (name, size).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareCreg { name, size } => Bytecode {
                    opcode: OpCode::DeclareCreg,
                    operands: (name, size).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareGate { name, num_qubits } => Bytecode {
                    opcode: OpCode::DeclareGate,
                    operands: (name, num_qubits).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::GateInBody {
                    id,
                    arguments,
                    qubits,
                } => Bytecode {
                    // In Python space, we don't have to be worried about the types of the
                    // parameters changing here, so we can just use `OpCode::Gate` unlike in the
                    // internal bytecode.
                    opcode: OpCode::Gate,
                    operands: (id, GateBodyArguments(arguments), qubits)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::EndDeclareGate {} => Bytecode {
                    opcode: OpCode::EndDeclareGate,
                    operands: ().into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareOpaque { name, num_qubits } => Bytecode {
                    opcode: OpCode::DeclareOpaque,
                    operands: (name, num_qubits).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::SpecialInclude { indices } => Bytecode {
                    opcode: OpCode::SpecialInclude,
                    operands: (indices,).into_pyobject(py)?.into_any().unbind(),
                },
            },
        )
    }
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
            self.parser_state.parse_next(&mut self.buffer, py)?;
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
