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

//! The Python binding of the op catalogue, one class per operation.
//!
//! Every class here extends `ProgramOp`, which holds the op and answers for it, so a class of the
//! catalogue is a constructor and a name.

use pyo3::PyClass;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::chain;
use super::tensor::{PyDType, parse_shape, tensor};
use crate::ops::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BoxedProgramOp, BroadcastTo, Cast,
    Constant, Divide, Mean, Multiply, Parity, Power, ProgramOp, Remainder, Std, Subtract, Variance,
};
use crate::tensor::TensorType;

/// One of the operations an instruction can perform, applied by adding an instruction to a program
/// function.
///
/// An op has whatever the operation needs beyond its operands, such as the axis a reduction folds
/// along, and it reports the types it produces from the types it is given. Each operation Qiskit
/// defines has a class of its own, and this is the base of all of them.
#[pyclass(
    name = "ProgramOp",
    module = "qiskit.quantum_program.ops",
    subclass,
    frozen
)]
pub struct PyProgramOp {
    pub(super) op: BoxedProgramOp,
}

#[pymethods]
impl PyProgramOp {
    /// The type name, qualified by its namespace.
    #[getter]
    fn full_name(&self) -> String {
        self.op.full_name()
    }

    /// The types this operation produces from operands of type ``operands``.
    ///
    /// Args:
    ///     operands: The type of each operand, one per operand the operation takes.
    ///
    /// Returns:
    ///     One type per value the operation produces.
    ///
    /// Raises:
    ///     ValueError: If there is not one type per operand, or if the operation does not accept the
    ///         types given.
    #[pyo3(signature = (operands, /))]
    fn output_types(&self, operands: Vec<TensorType>) -> PyResult<Vec<TensorType>> {
        let arity = self.op.arity();
        if operands.len() != arity {
            return Err(PyValueError::new_err(format!(
                "{} takes {arity} operands, got {}",
                self.op.full_name(),
                operands.len()
            )));
        }
        self.op.infer_output_types(&operands).map_err(|error| {
            PyValueError::new_err(format!("{}: {}", self.op.full_name(), chain(&*error)))
        })
    }
}

/// Initialize `class`, one of the classes of the catalogue, over the op `op`.
///
/// The op goes on the base class, so a class of the catalogue holds nothing of its own.
fn init<C, O>(op: O, class: C) -> PyClassInitializer<C>
where
    C: PyClass<BaseType = PyProgramOp>,
    O: ProgramOp + Clone + Send + Sync + 'static,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    PyClassInitializer::from(PyProgramOp { op: Box::new(op) }).add_subclass(class)
}

/// Define the class of an operation that has no payload of its own.
macro_rules! payload_free_op {
    ($(#[$documentation:meta])* $class:ident, $name:literal, $op:expr) => {
        $(#[$documentation])*
        #[pyclass(
            name = $name,
            module = "qiskit.quantum_program.ops",
            extends = PyProgramOp,
            frozen
        )]
        pub struct $class;

        #[pymethods]
        impl $class {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                init($op, Self)
            }
        }
    };
}

payload_free_op!(
    /// Add two tensors elementwise.
    ///
    /// The operands promote to a common dtype and broadcast to a common shape, as they do for every
    /// elementwise operation.
    PyAdd,
    "Add",
    Add
);

payload_free_op!(
    /// Subtract the second tensor from the first, elementwise.
    PySubtract,
    "Subtract",
    Subtract
);

payload_free_op!(
    /// Multiply two tensors elementwise.
    PyMultiply,
    "Multiply",
    Multiply
);

payload_free_op!(
    /// Divide the first tensor by the second, elementwise.
    ///
    /// A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.
    PyDivide,
    "Divide",
    Divide
);

payload_free_op!(
    /// Remainder of the first tensor divided by the second, elementwise.
    ///
    /// A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.
    PyRemainder,
    "Remainder",
    Remainder
);

payload_free_op!(
    /// Raise the first tensor to the power of the second, elementwise.
    PyPower,
    "Power",
    Power
);

payload_free_op!(
    /// Bitwise AND of two bit-valued tensors, elementwise.
    PyBitwiseAnd,
    "BitwiseAnd",
    BitwiseAnd
);

payload_free_op!(
    /// Bitwise OR of two bit-valued tensors, elementwise.
    PyBitwiseOr,
    "BitwiseOr",
    BitwiseOr
);

payload_free_op!(
    /// Bitwise XOR of two bit-valued tensors, elementwise.
    PyBitwiseXor,
    "BitwiseXor",
    BitwiseXor
);

payload_free_op!(
    /// Bitwise NOT of a bit-valued tensor, elementwise.
    PyBitwiseNot,
    "BitwiseNot",
    BitwiseNot
);

/// XOR-reduce the bits of a tensor along one axis, removing that axis.
///
/// The parity of a sequence of bits is 1 when an odd number of them are 1, and 0 otherwise.
///
/// Args:
///     axis: The axis to XOR-reduce along.
#[pyclass(
    name = "Parity",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyParity;

#[pymethods]
impl PyParity {
    #[new]
    #[pyo3(signature = (axis, /))]
    fn new(axis: usize) -> PyClassInitializer<Self> {
        init(Parity::new(axis), Self)
    }
}

/// Average a tensor along one axis, removing that axis.
///
/// Averaging bits or integers gives a float.
///
/// Args:
///     axis: The axis to average along.
#[pyclass(
    name = "Mean",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyMean;

#[pymethods]
impl PyMean {
    #[new]
    #[pyo3(signature = (axis, /))]
    fn new(axis: usize) -> PyClassInitializer<Self> {
        init(Mean::new(axis), Self)
    }
}

/// Variance of a tensor along one axis, removing that axis.
///
/// The sum of squared deviations is divided by ``n - ddof``, where ``n`` is the length of the axis.
///
/// Args:
///     axis: The axis to take the variance along.
///     ddof: The delta degrees of freedom, subtracted from the divisor.
#[pyclass(
    name = "Variance",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyVariance;

#[pymethods]
impl PyVariance {
    #[new]
    #[pyo3(signature = (axis, /, ddof = 0.0))]
    fn new(axis: usize, ddof: f64) -> PyClassInitializer<Self> {
        init(Variance::new(axis, ddof), Self)
    }
}

/// Standard deviation of a tensor along one axis, removing that axis.
///
/// This is the square root of the variance, and ``ddof`` means the same thing.
///
/// Args:
///     axis: The axis to take the standard deviation along.
///     ddof: The delta degrees of freedom, subtracted from the divisor.
#[pyclass(
    name = "Std",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyStd;

#[pymethods]
impl PyStd {
    #[new]
    #[pyo3(signature = (axis, /, ddof = 0.0))]
    fn new(axis: usize, ddof: f64) -> PyClassInitializer<Self> {
        init(Std::new(axis, ddof), Self)
    }
}

/// Reinterpret a tensor as another dtype, keeping its shape.
///
/// Args:
///     target: The dtype to cast to. A complex value cannot be cast to a real dtype.
#[pyclass(
    name = "Cast",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyCast;

#[pymethods]
impl PyCast {
    #[new]
    #[pyo3(signature = (target, /))]
    fn new(target: PyDType) -> PyClassInitializer<Self> {
        init(Cast::new(target.0), Self)
    }
}

/// Broadcast a tensor to a shape, aligning its axes with the trailing axes of that shape.
///
/// Args:
///     target: The shape to reach. Each axis of the operand must either match the axis it aligns
///         with or have size one.
#[pyclass(
    name = "BroadcastTo",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyBroadcastTo;

#[pymethods]
impl PyBroadcastTo {
    #[new]
    #[pyo3(signature = (target, /))]
    fn new(target: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        Ok(init(BroadcastTo::new(parse_shape(target)?), Self))
    }
}

/// Supply a tensor the program holds, rather than one given at call time.
///
/// Args:
///     value: The value to hold, read with :func:`numpy.asarray`.
#[pyclass(
    name = "Constant",
    module = "qiskit.quantum_program.ops",
    extends = PyProgramOp,
    frozen
)]
pub struct PyConstant;

#[pymethods]
impl PyConstant {
    #[new]
    #[pyo3(signature = (value, /))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        Ok(init(Constant::new(tensor(value)?), Self))
    }
}
