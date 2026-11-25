// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Bindings for the `qiskit.quantum_program` Python package.

mod data_tree;
mod ops;
mod program;
mod tensor;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::tensor::TensorType;
pub use data_tree::PyDataTree;
use ops::{
    PyAdd, PyBitwiseAnd, PyBitwiseNot, PyBitwiseOr, PyBitwiseXor, PyBroadcastTo, PyCast,
    PyConstant, PyDivide, PyMean, PyMultiply, PyParity, PyPower, PyProgramOp, PyRemainder, PyStd,
    PySubtract, PyVariance,
};
use program::{PyFunctionBuilder, PyQuantumProgram, PyValue};
pub use tensor::{PyBounded, PyDType};

/// Return `error` and everything that caused it, as one message.
///
/// Nothing sets a Python exception's cause here, so each source is appended to the message instead.
fn chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

/// `error` as a `ValueError`.
fn value_error(error: &dyn std::error::Error) -> PyErr {
    PyValueError::new_err(chain(error))
}

/// Register the `quantum_program` submodule.
pub fn quantum_program(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBounded>()?;
    m.add_class::<PyDataTree>()?;
    m.add_class::<PyDType>()?;
    m.add_class::<PyFunctionBuilder>()?;
    m.add_class::<PyQuantumProgram>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<TensorType>()?;
    // The op catalogue, whose base class must be registered before the classes extending it.
    m.add_class::<PyProgramOp>()?;
    m.add_class::<PyAdd>()?;
    m.add_class::<PyBitwiseAnd>()?;
    m.add_class::<PyBitwiseNot>()?;
    m.add_class::<PyBitwiseOr>()?;
    m.add_class::<PyBitwiseXor>()?;
    m.add_class::<PyBroadcastTo>()?;
    m.add_class::<PyCast>()?;
    m.add_class::<PyConstant>()?;
    m.add_class::<PyDivide>()?;
    m.add_class::<PyMean>()?;
    m.add_class::<PyMultiply>()?;
    m.add_class::<PyParity>()?;
    m.add_class::<PyPower>()?;
    m.add_class::<PyRemainder>()?;
    m.add_class::<PyStd>()?;
    m.add_class::<PySubtract>()?;
    m.add_class::<PyVariance>()?;
    Ok(())
}
