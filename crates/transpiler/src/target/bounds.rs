// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt;

use pyo3::prelude::*;
use pyo3::Python;

use super::errors::TargetError;
use qiskit_circuit::dag_circuit::DAGCircuit;
use smallvec::SmallVec;

#[derive(Clone)]
enum CallbackType {
    Python(PyObject),
    Native(fn(&[f64]) -> DAGCircuit),
}

impl CallbackType {
    fn call(&self, angles: &[f64]) -> PyResult<DAGCircuit> {
        match self {
            Self::Python(inner) => {
                Python::with_gil(|py| inner.bind(py).call1((angles,))?.extract())
            }
            Self::Native(inner) => Ok(inner(angles)),
        }
    }
}

#[derive(Clone)]
pub(crate) struct AngleBound {
    bounds: SmallVec<[Option<[f64; 2]>; 3]>,
    callback: CallbackType,
}

impl fmt::Debug for AngleBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AngleBound")
            .field("bounds", &self.bounds)
            .finish()
    }
}

impl AngleBound {
    pub fn new_native(
        bounds: SmallVec<[Option<[f64; 2]>; 3]>,
        callback: fn(&[f64]) -> DAGCircuit,
    ) -> Result<Self, TargetError> {
        for [low, high] in bounds.iter().flatten() {
            if low >= high {
                return Err(TargetError::InvalidBounds {
                    low: *low,
                    high: *high,
                });
            }
        }
        Ok(Self {
            bounds,
            callback: CallbackType::Native(callback),
        })
    }

    pub fn new_py(
        bounds: SmallVec<[Option<[f64; 2]>; 3]>,
        callback: PyObject,
    ) -> Result<Self, TargetError> {
        for [low, high] in bounds.iter().flatten() {
            if low >= high {
                return Err(TargetError::InvalidBounds {
                    low: *low,
                    high: *high,
                });
            }
        }
        Ok(Self {
            bounds,
            callback: CallbackType::Python(callback),
        })
    }

    pub fn angles_supported(&self, angles: &[f64]) -> bool {
        angles
            .iter()
            .zip(&self.bounds)
            .all(|(angle, bound)| match bound {
                Some([low, high]) => !(angle < low || angle > high),
                None => true,
            })
    }

    pub fn get_replacement_circuit(&self, angle: &[f64]) -> PyResult<DAGCircuit> {
        self.callback.call(angle)
    }
}
