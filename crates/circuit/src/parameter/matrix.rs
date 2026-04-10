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


use ndarray::{Array0, Array2, Zip};
use ndarray::parallel::prelude::*;
use ndarray::linalg::Dot;


use crate::parameter::parameter_expression::ParameterExpression;
use crate::parameter::vector::VectorExpression;



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

    /// resize the matrix
    pub fn resize(&mut self, nrow: usize, ncol: usize) {
        self.elements = self.elements.into_shape_with_order((nrow, ncol)).unwrap()
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
    /// divide a matrix by a scalar and assign
    pub fn _div_scalar_assign_par(&mut self, rhs: &ParameterExpression) {
        self.elements.par_map_inplace(|l| *l = &*l / rhs);
    }

    /// multiply a vector
    pub fn _mul_vec(&self, rhs: &VectorExpression) -> VectorExpression {
        VectorExpression {
            elements: self.elements.dot(&rhs.elements),
        }
    }

    /// multiply a matrix
    pub fn _mul(&self, rhs: &MatrixExpression) -> MatrixExpression {
        MatrixExpression {
            elements: self.elements.dot(&rhs.elements),
        }
    }
    pub fn _mul_par(&self, rhs: &MatrixExpression) -> MatrixExpression {
        MatrixExpression {
            elements: &self.elements * &rhs.elements,
        }
    }

}





