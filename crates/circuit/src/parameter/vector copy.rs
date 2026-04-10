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


use crate::parameter::parameter_expression::{ParameterExpression, ParameterError, PyParameterExpression};
use pyo3::types::{PyString};

use pyo3::prelude::*;



/// A vector of parameter expression.
///
///
#[derive(Clone, Debug)]
pub struct VectorExpression {
    pub(crate) elements: Vec<ParameterExpression>,
}

impl Default for VectorExpression {
    /// The default constructor returns zero.
    fn default() -> Self {
        Self {
            elements: Vec::<ParameterExpression>::new(),
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
            elements: vec![ParameterExpression::from_f64(value); ndim],
        }
    }

    /// get dimension of the vector
    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    /// resize the vector
    pub fn resize(&mut self, size: usize) {
        self.elements.resize(size, ParameterExpression::from_f64(0.0f64));
    }

    /// add vectors
    pub fn add(&self, rhs: &VectorExpression) -> Result<Self, ParameterError> {
        Ok(Self {
            elements: self.elements.iter().zip(rhs.elements.iter())
                        .filter_map(|(l, r)| l.add(r).ok()).collect::<Vec<ParameterExpression>>()
        })
    }
    /// add and assign vectors
    pub fn adda(&mut self, rhs: &VectorExpression) -> Result<&Self, ParameterError> {
        self.elements.iter_mut().zip(rhs.elements.iter())
            .for_each(|(l, r)| {
                match l.add(r) {
                    Ok(o) => *l = o,
                    Err(_) => (),
                }});
        Ok(self)
    }

    /// sub vectors
    pub fn sub(&self, rhs: &VectorExpression) -> Result<Self, ParameterError> {
        Ok(Self {
            elements: self.elements.iter().zip(rhs.elements.iter())
                        .filter_map(|(l, r)| l.sub(r).ok()).collect::<Vec<ParameterExpression>>()
        })
    }

    /// sub and assign vectors
    pub fn suba(&mut self, rhs: &VectorExpression) -> Result<&Self, ParameterError> {
        self.elements.iter_mut().zip(rhs.elements.iter())
            .for_each(|(l, r)| {
                match l.sub(r) {
                    Ok(o) => *l = o,
                    Err(_) => (),
                }});
        Ok(self)
    }

    /// multiply a scalar to a vector
    pub fn mul(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        Ok(Self {
            elements: self.elements.iter()
                        .filter_map(|l| l.mul(rhs).ok()).collect::<Vec<ParameterExpression>>()
        })
    }

    /// multiply a scalar to a vector and assign
    pub fn mula(&mut self, rhs: &ParameterExpression) -> Result<&Self, ParameterError> {
        self.elements.iter_mut()
            .for_each(|l| {
                match l.mul(rhs) {
                    Ok(o) => *l = o,
                    Err(_) => (),
                }});
        Ok(self)
    }

    /// divide a vector by a scalar
    pub fn div(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        Ok(Self {
            elements: self.elements.iter()
                        .filter_map(|l| l.div(rhs).ok()).collect::<Vec<ParameterExpression>>()
        })
    }

    /// divide a vector by a scalar and assign
    pub fn diva(&mut self, rhs: &ParameterExpression) -> Result<&Self, ParameterError> {
        self.elements.iter_mut()
            .for_each(|l| {
                match l.div(rhs) {
                    Ok(o) => *l = o,
                    Err(_) => (),
                }});
        Ok(self)
    }

    /// calculate dot product
    pub fn dot(&self, rhs: &VectorExpression) -> Result<ParameterExpression, ParameterError> {
        let size = cmp::min(self.dim(), rhs.dim());
        let mut out = ParameterExpression::from_f64(0.0f64);
        for i in 0..size {
            out = out.add(&self.elements[i].mul(&rhs.elements[i])?)?;
        }
        Ok(out)
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
        if let Ok(a) = self.inner.add(&rhs.inner) {
            Ok(Self{ inner: a})
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                            "Unsupported data type for __add__",
            ))
        }
    }


}


