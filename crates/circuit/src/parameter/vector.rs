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

use std::cmp;
use std::ops::{Index, IndexMut};
use std::fmt;



use ndarray::{Array0, Array1, ArrayView0, Axis};
use ndarray::linalg::Dot;
use ndarray::Zip;
use ndarray::parallel::prelude::*;

use crate::parameter::parameter_expression::{ParameterExpression, PyParameterExpression};
use pyo3::types::{PyString};

use pyo3::prelude::*;



/// A vector of parameter expression.
///
///
#[derive(Clone, Debug)]
pub struct VectorExpression {
    pub(crate) elements: Array1<ParameterExpression>,
}

impl Default for VectorExpression {
    fn default() -> Self {
        Self {
            elements: Array1::<ParameterExpression>::default((0)),
        }
    }
}


impl Index<usize> for VectorExpression {
    type Output = ParameterExpression;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl IndexMut<usize> for VectorExpression {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl fmt::Display for VectorExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.dim();
        write!(f, "[")?;
        self.elements.iter().enumerate().try_for_each(|(i, p)| if i == n - 1 {write!(f, "{}]", p)} else {write!(f, "{}, ", p)})
    }
}

impl VectorExpression {
    /// Initialize a vector with its size and initial values
    pub fn new(ndim: usize, value: f64) -> Self {
        Self {
            elements: Array1::<ParameterExpression>::from_elem((ndim), ParameterExpression::from_f64(value)),
        }
    }

    /// get dimension of the vector
    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    /// resize the vector
    pub fn resize(&mut self, size: usize) {
        if size >= self.elements.len() {
            let a = Array0::from_elem((), ParameterExpression::from_f64(0.0f64));
            for _ in self.elements.len()..size {
                self.elements.push(Axis(0), ArrayView0::from(&a));
            }
        } else {
            self.elements = self.elements.slice(ndarray::s![0..size]).to_owned();
        }
    }

    /// add vectors
    pub fn _add(&self, rhs: &VectorExpression) -> Self {
        Self {
            elements: &self.elements + &rhs.elements,
        }
    }
    /// add and assign vectors
    pub fn _add_assign(&mut self, rhs: &VectorExpression) {
        self.elements += &rhs.elements;
    }

    /// add vectors
    pub fn _add_par(&self, rhs: &VectorExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).and(&rhs.elements).par_map_collect(|l, r| l + r),
        }
    }
    /// add and assign vectors
    pub fn _add_assign_par(&mut self, rhs: &VectorExpression) {
        Zip::from(&mut self.elements).and(&rhs.elements).par_for_each(|l, r| *l = &*l + r);
    }

    /// sub vectors
    pub fn _sub(&self, rhs: &VectorExpression) -> Self {
        Self {
            elements: &self.elements - &rhs.elements,
        }
    }
    /// sub and assign vectors
    pub fn _sub_assign(&mut self, rhs: &VectorExpression) {
        self.elements -= &rhs.elements;
    }

    /// sub vectors
    pub fn _sub_par(&self, rhs: &VectorExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).and(&rhs.elements).par_map_collect(|l, r| l - r),
        }
    }
    /// add and assign vectors
    pub fn _sub_assign_par(&mut self, rhs: &VectorExpression) {
        Zip::from(&mut self.elements).and(&rhs.elements).par_for_each(|l, r| *l = &*l - r);
    }

    /// multiply a scalar to a vector
    pub fn _mul(&self, rhs: &ParameterExpression) -> Self {
        let rhs = Array0::from_elem((), rhs.clone());
        Self {
            elements: &self.elements * rhs,
        }
    }
    /// multiply a scalar to a vector and assign
    pub fn _mul_assign(&mut self, rhs: &ParameterExpression) {
        let rhs = Array0::from_elem((), rhs.clone());
        self.elements *= &rhs;
    }

    /// multiply a scalar to a vector
    pub fn _mul_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| l * rhs),
        }
    }
    /// multiply a scalar to a vector and assign
    pub fn _mul_assign_par(&mut self, rhs: &ParameterExpression) {
        self.elements.par_map_inplace(|l| *l = &*l * rhs);
    }

    /// divide a vector by a scalar
    pub fn _div(&self, rhs: &ParameterExpression) -> Self {
        let rhs = Array0::from_elem((), rhs.clone());
        Self {
            elements: &self.elements / rhs,
        }
    }
    /// divide a vector by a scalar and assign
    pub fn _div_assign(&mut self, rhs: &ParameterExpression) {
        let rhs = Array0::from_elem((), rhs.clone());
        self.elements /= &rhs;
    }

    /// divide a vector by a scalar
    pub fn _div_par(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: Zip::from(&self.elements).par_map_collect(|l| l / rhs),
        }
    }
    /// divide a vector by a scalar and assign
    pub fn _div_assign_par(&mut self, rhs: &ParameterExpression) {
        self.elements.par_map_inplace(|l| *l = &*l / rhs);
    }

    /// calculate dot product
    pub fn dot(&self, rhs: &VectorExpression) -> ParameterExpression {
        self.elements.dot(&rhs.elements)
    }
    /// calculate dot product
    pub fn dot_par(&self, rhs: &VectorExpression) -> ParameterExpression {
        Zip::from(&self.elements).and(&rhs.elements).par_fold(||ParameterExpression::from_f64(0.0f64), |sum, l, r| sum + l * r, |sum, t| sum + t)
    }

}



#[pyclass(
    subclass,
    module = "qiskit._accelerate.circuit",
    name = "VectorExpression",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyVectorExpression {
    pub inner: VectorExpression,
}


#[pymethods]
impl PyVectorExpression {
    #[new]
    #[pyo3(signature = (ndim=1, value=0.0f64))]
    pub fn py_new(
        ndim: usize,
        value: Option<f64>,
    ) -> PyResult<Self> {
        let value = match value {
            Some(v) => v,
            None => 0.0f64,
        };
        Ok(Self{ inner: VectorExpression::new(ndim, value) })
    }

    /// return size of the vector
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!("VectorExpression({})", self.inner);
        PyString::new(py, str.as_str())
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub fn __getitem__(&self, index: usize) -> PyResult<PyParameterExpression> {
        if index >= self.inner.dim() {
            Err(pyo3::exceptions::PyIndexError::new_err(" VectorExpression::__getitem__ : Error index out of bounds"))
        } else {
            Ok(PyParameterExpression{inner: self.inner.elements[index].clone()})
        }
    }

    pub fn __setitem__(&mut self, index: usize, value: &Bound<PyAny>) -> PyResult<()> {
        if index >= self.inner.dim() {
            Err(pyo3::exceptions::PyIndexError::new_err(" VectorExpression__setitem__ : Error index out of bounds"))
        } else {
            match PyParameterExpression::extract_coerce(value.as_borrowed()) {
                Ok(value) => {
                    self.inner.elements[index] = value.inner;
                    Ok(())
                }
                Err(err) => Err(err)
            }
        }
    }

    pub fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {inner: self.inner._add(&rhs.inner)})
    }

    pub fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {inner: self.inner._sub(&rhs.inner)})
    }

}


