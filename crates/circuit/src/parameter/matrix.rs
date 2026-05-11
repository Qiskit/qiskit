// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt;

use ndarray::{Array0, Array2, Zip, Axis};
use ndarray::parallel::prelude::*;
//use ndarray::linalg::Dot;
use numpy::PyArrayMethods;
use numpy::array::PyArray2;

use hashbrown::HashMap;

use crate::parameter::symbol_expr::SymbolExpr;
use crate::parameter::parameter_expression::{ParameterExpression, PyParameterExpression};
use super::symbol_expr::Value;
use crate::parameter::vector::{PyVectorExpression, VectorExpression};


use pyo3::types::{PyString, PyNotImplemented, PyTuple};
use pyo3::{IntoPyObjectExt, prelude::*};
use pyo3::exceptions::{PyIndexError, PyTypeError};

/// A matrix of parameter expression.
///
///
#[derive(Clone, Debug)]
pub struct MatrixExpression {
    pub(crate) elements: Array2<ParameterExpression>,
}

impl Default for MatrixExpression {
    fn default() -> Self {
        Self {
            elements: Array2::<ParameterExpression>::default((0, 0)),
        }
    }
}

impl fmt::Display for MatrixExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.elements)
    }
}


impl MatrixExpression {
    /// Initialize a matrix with its size and initial values
    pub fn new(nrow: usize, ncol: usize, value: f64) -> Self {
        Self {
            elements : Array2::from_elem((nrow, ncol), ParameterExpression::from_f64(value)),
        }
    }

    /// get number of rows of the matrix
    pub fn nrows(&self) -> usize {
        self.elements.nrows()
    }

    /// get number of columns of the matrix
    pub fn ncols(&self) -> usize {
        self.elements.ncols()
    }


    /// get element
    pub fn get(&self, row: usize, col: usize) -> ParameterExpression {
        self.elements[[row, col]].clone()
    }

    /// set element
    pub fn set(&mut self, row: usize, col: usize, val: &ParameterExpression) {
        self.elements[[row, col]] = val.clone();
    }

    /// resize the matrix
    pub fn resize(&mut self, nrow: usize, ncol: usize) {
        self.elements = self.elements.clone().into_shape_with_order((nrow, ncol)).unwrap()
    }

    /// add matrices
    pub fn _add(&self, rhs: &MatrixExpression) -> Self {
        Self {
            elements: &self.elements + &rhs.elements,
        }
    }
    /// add and assign matrices
    pub fn _add_assign(&mut self, rhs: &MatrixExpression) {
        self.elements += &rhs.elements;
    }
    /// add matrices
    pub fn _add_par(&self, rhs: &MatrixExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).and(&rhs.elements).par_map_collect(|l, r| l + r),
        }
    }
    /// add and assign matrices
    pub fn _add_assign_par(&mut self, rhs: &MatrixExpression) {
        Zip::from(&mut self.elements).and(&rhs.elements).par_for_each(|l, r| *l = &*l + r);
    }

    /// sub matrices
    pub fn _sub(&self, rhs: &MatrixExpression) -> Self {
        Self {
            elements: &self.elements - &rhs.elements,
        }
    }
    /// sub and assign matrices
    pub fn _sub_assign(&mut self, rhs: &MatrixExpression) {
        self.elements -= &rhs.elements;
    }
    /// sub matrices
    pub fn _sub_par(&self, rhs: &MatrixExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).and(&rhs.elements).par_map_collect(|l, r| l - r),
        }
    }
    /// sub and assign matrices
    pub fn _sub_assign_par(&mut self, rhs: &MatrixExpression) {
        Zip::from(&mut self.elements).and(&rhs.elements).par_for_each(|l, r| *l = &*l - r);
    }

    /// multiply a scalar to a matrix
    pub fn _mul_scalar(&self, rhs: &ParameterExpression) -> Self {
        let rhs = Array0::from_elem((), rhs.clone());
        Self {
            elements: &self.elements * rhs,
        }
    }
    /// multiply a scalar to a matrix and assign
    pub fn _mul_scalar_assign(&mut self, rhs: &ParameterExpression) {
        let rhs = Array0::from_elem((), rhs.clone());
        self.elements *= &rhs;
    }

    /// multiply a scalar to a matrix
    pub fn _mul_scalar_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| l * rhs),
        }
    }

    /// multiply a scalar to a matrix
    pub fn _rmul_scalar_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| rhs * l),
        }
    }

    /// multiply a scalar to a matrix and assign
    pub fn _mul_scalar_assign_par(&mut self, rhs: &ParameterExpression) {
        self.elements.par_map_inplace(|l| *l = &*l * rhs);
    }

    /// divide a matrix by a scalar
    pub fn _div_scalar(&self, rhs: &ParameterExpression) -> Self {
        let rhs = Array0::from_elem((), rhs.clone());
        Self {
            elements: &self.elements / rhs,
        }
    }
    /// divide a matrix by a scalar and assign
    pub fn _div_scalar_assign(&mut self, rhs: &ParameterExpression) {
        let rhs = Array0::from_elem((), rhs.clone());
        self.elements /= &rhs;
    }

    /// divide a matrix by a scalar
    pub fn _div_scalar_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| l / rhs),
        }
    }
    pub fn _rdiv_scalar_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| rhs / l),
        }
    }

    /// divide a matrix by a scalar and assign
    pub fn _div_scalar_assign_par(&mut self, rhs: &ParameterExpression) {
        self.elements.par_map_inplace(|l| *l = &*l / rhs);
    }

    /// multiply a vector
    pub fn _mul_vec(&self, rhs: &VectorExpression) -> VectorExpression {
        VectorExpression {
            elements:
                self.elements.rows().into_iter().
                    map(|row| row.iter().zip(rhs.elements.iter()).fold(ParameterExpression::from_f64(0.0f64), |sum, (l, r)| sum + l * r)).collect()
        }
    }

    pub fn _mul_vec_par(&self, rhs: &VectorExpression) -> VectorExpression {
        VectorExpression {
            elements:
                Zip::from(self.elements.axis_iter(Axis(0))).
                    par_map_collect(|row| row.iter().zip(rhs.elements.iter()).fold(ParameterExpression::from_f64(0.0f64), |sum, (l, r)| sum + l * r))
        }
    }

    /// multiply a matrix
    pub fn _mul(&self, rhs: &MatrixExpression) -> MatrixExpression {
        let mut out = Array2::<ParameterExpression>::from_elem(self.elements.dim(), ParameterExpression::from_f64(0.0f64));
        out.axis_iter_mut(Axis(0)).enumerate().for_each(|(i, row)|
                row.into_iter().enumerate().for_each(|(j, o)|
                    self.elements.index_axis(Axis(0), i).iter().zip(rhs.elements.index_axis(Axis(1), j).iter()).
                            for_each(|(l, r)| *o += l * r)));
        MatrixExpression {
            elements: out
        }
    }
    pub fn _mul_par(&self, rhs: &MatrixExpression) -> MatrixExpression {
        let mut out = Array2::<ParameterExpression>::from_elem(self.elements.dim(), ParameterExpression::from_f64(0.0f64));
        out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, row)|
                row.into_iter().enumerate().for_each(|(j, o)|
                    self.elements.index_axis(Axis(0), i).iter().zip(rhs.elements.index_axis(Axis(1), j).iter()).
                            for_each(|(l, r)| *o += l * r)));
        MatrixExpression {
            elements: out
        }
    }

}


#[pyclass(
    subclass,
    module = "qiskit._accelerate.circuit",
    name = "MatrixExpression",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyMatrixExpression {
    pub inner: MatrixExpression,
}

impl From<MatrixExpression> for PyMatrixExpression {
    fn from(value: MatrixExpression) -> Self {
        Self { inner: value }
    }
}

impl PyMatrixExpression {
    /// Attempt to extract a `PyMatrixExpression` from a bound `PyAny`.
    ///
    /// This will try to coerce to the strictest data type:
    ///
    /// # Arguments:
    ///
    /// * ob - The bound `PyAny` to extract from.
    ///
    /// # Returns
    ///
    /// * `Ok(PyVectorExpression)` - The extracted expression.
    /// * `Err(PyResult)` - An error if extraction to all above types failed.
    pub fn extract_coerce(ob: Borrowed<PyAny>) -> PyResult<PyMatrixExpression> {
        if let Ok(matrix) = ob.cast::<PyArray2<i64>>() {
            let matrix  = matrix.try_readonly()?;
            let matrix = matrix.as_array();
            Ok(MatrixExpression{ elements : Array2::<ParameterExpression>::from_shape_fn(matrix.dim(), |(r, c)| ParameterExpression::new(SymbolExpr::Value(Value::from(matrix[[r, c]])), HashMap::new())) }.into())
        } else if let Ok(matrix) = ob.cast::<PyArray2<f64>>() {
            let matrix  = matrix.try_readonly()?;
            let matrix = matrix.as_array();
            Ok(MatrixExpression{ elements : Array2::<ParameterExpression>::from_shape_fn(matrix.dim(), |(r, c)| ParameterExpression::from_f64(matrix[[r, c]])) }.into())
        } else {
            ob.extract::<PyMatrixExpression>().map_err(Into::into)
        }
    }
}

#[pymethods]
impl PyMatrixExpression {
    #[new]
    #[pyo3(signature = (nr=1, nc=1, value=0.0f64))]
    pub fn py_new(
        nr: usize,
        nc: usize,
        value: Option<f64>,
    ) -> PyResult<Self> {
        let value = match value {
            Some(v) => v,
            None => 0.0f64,
        };
        Ok(Self{ inner: MatrixExpression::new(nr, nc, value) })
    }

    /// return number of rows
    #[getter]
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// return number of columns
    #[getter]
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!("MatrixExpression({})", self.inner);
        PyString::new(py, str.as_str())
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }


    pub fn __getitem__<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(pos) = ob.cast::<PyTuple>() {
            if pos.len() != 2 {
                return Err(PyIndexError::new_err("matrix dimension must be 2"));
            }
            if let (Ok(r), Ok(c)) = (pos.get_item(0)?.extract::<usize>(), pos.get_item(1)?.extract::<usize>()) {
                PyParameterExpression::from(self.inner.elements[[r, c]].clone()).into_bound_py_any(ob.py())
            } else {
                Err(PyTypeError::new_err("index must be tuple of int"))
            }
        } else {
            Err(PyTypeError::new_err("index must be tuple of int"))
        }
    }

    pub fn __setitem__<'py>(&mut self, ob: Bound<'py, PyAny>, value: Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(pos) = ob.cast::<PyTuple>() {
            if pos.len() != 2 {
                return Err(PyIndexError::new_err("matrix dimension must be 2"));
            }
            if let (Ok(r), Ok(c)) = (pos.get_item(0)?.extract::<usize>(), pos.get_item(1)?.extract::<usize>()) {
                if let Ok(value) = PyParameterExpression::extract_coerce(value.as_borrowed()) {
                    self.inner.elements[[r, c]] = value.inner;
                    Ok(())
                } else {
                    Err(PyTypeError::new_err("unsupported type for Matrix element"))
                }
            } else {
                Err(PyTypeError::new_err("index must be tuple of int"))
            }
        } else {
            Err(PyTypeError::new_err("index must be tuple of int"))
        }
    }

    pub fn __add__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._add_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __radd__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(rhs.inner._add_par(&self.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __sub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._sub_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __rsub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(rhs.inner._sub_par(&self.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._mul_par(&rhs.inner)).into_bound_py_any(py)
        } else if let Ok(rhs) = PyVectorExpression::extract_coerce(rhs.as_borrowed()) {
            PyVectorExpression::from(self.inner._mul_vec_par(&rhs.inner)).into_bound_py_any(py)
        } else if let Ok(rhs) = PyParameterExpression::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._mul_scalar_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __rmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(rhs.inner._mul_par(&self.inner)).into_bound_py_any(py)
        } else if let Ok(rhs) = PyParameterExpression::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._rmul_scalar_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __truediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = PyParameterExpression::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._div_scalar_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __rtruediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = PyParameterExpression::extract_coerce(rhs.as_borrowed()) {
            PyMatrixExpression::from(self.inner._rdiv_scalar_par(&rhs.inner)).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }
}






