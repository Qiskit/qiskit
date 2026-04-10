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

use std::ops::{Index, IndexMut};
use std::fmt;



use nalgebra::DVector;

use crate::parameter::parameter_expression::{ParameterExpression, PyParameterExpression};
use pyo3::types::{PyString};

use pyo3::prelude::*;



/// A vector of parameter expression.
///
///
#[derive(Clone, Debug)]
pub struct VectorExpression {
    pub(crate) elements: DVector<ParameterExpression>,
}

impl Default for VectorExpression {
    fn default() -> Self {
        Self {
            elements: DVector::<ParameterExpression>::from_element(0, ParameterExpression::from_f64(0.0f64)),
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
            elements: DVector::<ParameterExpression>::from_element(ndim, ParameterExpression::from_f64(value)),
        }
    }

    /// get dimension of the vector
    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    /// resize the vector
    pub fn resize(&mut self, size: usize) {
        if size >= self.elements.len() {
            self.elements = &DVector::<ParameterExpression>::from_element(size, ParameterExpression::from_f64(0.0f64)) + &self.elements;
        } else {
            self.elements = DVector::<ParameterExpression>::from_iterator(size, (0..size).into_iter().map(|i| self.elements[i].clone()));
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

    /// multiply a scalar to a vector
    pub fn _mul(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: &self.elements * rhs.clone(),
        }
    }

    /// multiply a scalar to a vector and assign
    pub fn _mul_assign(&mut self, rhs: &ParameterExpression) {
        self.elements *= rhs.clone();
    }

    /// divide a vector by a scalar
    pub fn _div(&self, rhs: &ParameterExpression) -> Self {
        Self {
            elements: &self.elements / rhs.clone(),
        }
    }

    /// divide a vector by a scalar and assign
    pub fn _div_assign(&mut self, rhs: &ParameterExpression) {
        self.elements /= rhs.clone();
    }

    /// calculate dot product
    pub fn dot(&self, rhs: &VectorExpression) -> ParameterExpression {
        self.elements.dot(&rhs.elements)
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

    pub fn __add__(&self, rhs: &Self) -> Self {
        Self{ inner: self.inner._add(&rhs.inner), }
    }


}


