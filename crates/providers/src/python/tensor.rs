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

//! The Python binding of the tensor domain: dtypes, axes, tensor types and tensor values.
//!
//! A tensor value is read from anything `numpy.asarray` accepts, copying the buffer, and a `Bit`
//! tensor is read from a NumPy array of `bool`.
//! A tensor crosses the boundary as a NumPy array. Both directions copy the buffer, and a `Bit`
//! tensor is a NumPy array of `bool`.

use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyTuple};

use crate::tensor::{DType, Dim, Tensor, TensorType};

/// The element type of a tensor.
#[pyclass(
    name = "DType",
    module = "qiskit.quantum_program",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyDType(pub(super) DType);

#[pymethods]
#[allow(non_upper_case_globals)]
impl PyDType {
    #[classattr]
    const Bit: Self = Self(DType::Bit);
    #[classattr]
    const U8: Self = Self(DType::U8);
    #[classattr]
    const U16: Self = Self(DType::U16);
    #[classattr]
    const U32: Self = Self(DType::U32);
    #[classattr]
    const U64: Self = Self(DType::U64);
    #[classattr]
    const I8: Self = Self(DType::I8);
    #[classattr]
    const I16: Self = Self(DType::I16);
    #[classattr]
    const I32: Self = Self(DType::I32);
    #[classattr]
    const I64: Self = Self(DType::I64);
    #[classattr]
    const F32: Self = Self(DType::F32);
    #[classattr]
    const F64: Self = Self(DType::F64);
    #[classattr]
    const C64: Self = Self(DType::C64);
    #[classattr]
    const C128: Self = Self(DType::C128);

    /// Return a ``TensorType`` of this dtype whose shape is ``shape``.
    fn __getitem__(&self, shape: &Bound<'_, PyAny>) -> PyResult<TensorType> {
        let shape = match shape.cast::<PyTuple>() {
            Ok(axes) => axes
                .iter()
                .map(|axis| dim(&axis))
                .collect::<PyResult<_>>()?,
            Err(_) => vec![dim(shape)?],
        };
        Ok(TensorType {
            dtype: self.0,
            shape,
        })
    }

    fn __repr__(&self) -> String {
        format!("DType.{}", self.0)
    }
}

/// An axis whose size is not known until run time, but is provably at most `max`.
#[pyclass(name = "bounded", module = "qiskit.quantum_program", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
pub struct PyBounded {
    /// The largest size this axis can have.
    #[pyo3(get)]
    max: usize,
}

#[pymethods]
impl PyBounded {
    #[new]
    #[pyo3(signature = (max, /))]
    fn new(max: usize) -> Self {
        Self { max }
    }

    fn __repr__(&self) -> String {
        format!("bounded({})", self.max)
    }
}

#[pymethods]
impl TensorType {
    #[new]
    #[pyo3(signature = (dtype, shape, /))]
    fn new(dtype: PyDType, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            dtype: dtype.0,
            shape: parse_shape(shape)?,
        })
    }

    /// The element type.
    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType(self.dtype)
    }

    /// The size of each axis, an integer or a `bounded()`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let shape = self
            .shape
            .iter()
            .map(|&dim| dim_object(py, dim))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, shape)
    }

    fn __repr__(&self) -> String {
        format!("TensorType({self})")
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

/// The shape `object` describes, taking each of its items as one axis.
pub(super) fn parse_shape(object: &Bound<'_, PyAny>) -> PyResult<Vec<Dim>> {
    object.try_iter()?.map(|axis| dim(&axis?)).collect()
}

/// Return the dimension `object` describes: an integer size, or a `bounded()` axis.
fn dim(object: &Bound<'_, PyAny>) -> PyResult<Dim> {
    if let Ok(bounded) = object.cast::<PyBounded>() {
        return Ok(Dim::Bounded {
            max: bounded.get().max,
        });
    }
    if let Ok(size) = object.extract::<usize>() {
        return Ok(Dim::Fixed(size));
    }
    if object.cast::<PyInt>().is_ok() {
        return Err(PyValueError::new_err(format!(
            "an axis cannot have size {}",
            object.str()?
        )));
    }
    Err(PyTypeError::new_err(format!(
        "an axis is sized by an integer or by bounded(), not by {}",
        object.get_type().name()?
    )))
}

/// Return `dim` as an integer for a fixed axis or a `bounded()` for a bounded one.
fn dim_object(py: Python<'_>, dim: Dim) -> PyResult<Py<PyAny>> {
    match dim {
        Dim::Fixed(size) => Ok(size.into_pyobject(py)?.into_any().unbind()),
        Dim::Bounded { max } => Ok(Py::new(py, PyBounded { max })?.into_any()),
    }
}

/// Read `object` as a tensor, converting it with `numpy.asarray` first.
///
/// The dtype of the resulting array is the dtype of the tensor: nothing is converted to a dtype a
/// caller has not asked for.
pub(super) fn tensor(object: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let py = object.py();
    let array = py
        .import("numpy")?
        .call_method1("asarray", (object,))?
        .into_any();

    // A `Bit` tensor is stored as `u8`, which numpy spells `bool`.
    if let Ok(bits) = array.cast::<PyArrayDyn<bool>>() {
        let bits = bits.readonly().as_array().mapv(u8::from);
        return Ok(Tensor::Bit(bits.into_shared()));
    }
    macro_rules! read {
        ($($variant:ident($element:ty)),* $(,)?) => {
            $(
                if let Ok(array) = array.cast::<PyArrayDyn<$element>>() {
                    let array: ArrayD<$element> = array.readonly().as_array().to_owned();
                    return Ok(Tensor::$variant(array.into_shared()));
                }
            )*
        };
    }
    read!(
        C64(Complex32),
        C128(Complex64),
        F32(f32),
        F64(f64),
        I8(i8),
        I16(i16),
        I32(i32),
        I64(i64),
        U8(u8),
        U16(u16),
        U32(u32),
        U64(u64),
    );
    Err(PyTypeError::new_err(format!(
        "a tensor cannot hold numpy dtype {}",
        array.getattr("dtype")?
    )))
}
