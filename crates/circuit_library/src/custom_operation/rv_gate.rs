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

use std::{ops::Div, str::FromStr};

use ndarray::{Array1, array};
use numpy::Complex64;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::operations::{PyCustomOperation, ToPythonOperation};
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
};

use qiskit_quantum_info::QiskitError;
use qiskit_synthesis::euler_one_qubit_decomposer::{EulerBasis, EulerBasisSet, unitary_to_circuit};
use qiskit_util::{complex::c64, py::ImportOnceCell};
use smallvec::smallvec;

static RVGATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library.generalized_gates.rv", "RVGate");

#[derive(Debug, Clone, PartialEq)]
pub struct RVGate {
    basis_set: EulerBasisSet,
}

impl RVGate {
    pub fn new(euler_basis: EulerBasis) -> Self {
        let mut basis_set = EulerBasisSet::new();
        basis_set.add_basis(euler_basis);
        Self { basis_set }
    }

    pub fn basis(&self) -> EulerBasis {
        self.basis_set
            .get_bases()
            .next()
            .expect("The basis set within is alwyas of length 1")
    }
}

impl Default for RVGate {
    fn default() -> Self {
        let mut basis_set = EulerBasisSet::new();
        basis_set.add_basis(EulerBasis::U);
        Self { basis_set }
    }
}

impl Operation for RVGate {
    fn name(&self) -> &str {
        "rv"
    }

    fn num_qubits(&self) -> u32 {
        1
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        3
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for RVGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn inverse(
        &self,
        params: &[qiskit_circuit::operations::Param],
    ) -> Option<(
        qiskit_circuit::packed_instruction::PackedOperation,
        smallvec::SmallVec<[qiskit_circuit::operations::Param; 3]>,
    )> {
        let [Param::Float(vx), Param::Float(vy), Param::Float(vz)] = params else {
            return None;
        };
        Some((
            PackedOperation::from_custom_operation(Box::new(Self::default())),
            smallvec![Param::Float(-vx), Param::Float(-vy), Param::Float(-vz)],
        ))
    }

    fn matrix(
        &self,
        params: &[Param],
    ) -> Result<Option<ndarray::prelude::Array2<numpy::Complex64>>, Box<dyn std::error::Error>>
    {
        let [Param::Float(vx), Param::Float(vy), Param::Float(vz)] = params else {
            return Ok(None);
        };

        let array: Array1<Complex64> = array![
            Complex64::from(*vx),
            Complex64::from(*vy),
            Complex64::from(*vz)
        ];
        let angle = array.dot(&array).sqrt();
        if angle == Complex64::ZERO {
            return Ok(Some(array![
                [Complex64::ONE, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]));
        }
        let div = array.div(angle);

        let nx = div[0];
        let ny = div[1];
        let nz = div[2];

        let sin = (angle / Complex64::from(2.0)).sin();
        let cos = (angle / Complex64::from(2.0)).cos();

        Ok(Some(array![
            [
                cos - c64(0.0, 1.0) * nz * sin,
                (-ny - c64(0.0, 1.0) * nx) * sin
            ],
            [
                (ny - c64(0.0, 1.0) * nx) * sin,
                cos + c64(0.0, 1.0) * nz * sin
            ]
        ]))
    }

    fn definition(&self, params: &[Param]) -> Option<qiskit_circuit::circuit_data::CircuitData> {
        let Ok(Some(matrix)) = self.matrix(params) else {
            return None;
        };

        unitary_to_circuit(matrix.view(), &self.basis_set, 0, None, true, None)
            .expect("Circuit building should not fail")
    }
}

impl ToPythonOperation for RVGate {
    fn create_py_op<'py>(
        &self,
        py: Python<'py>,
        params: Option<smallvec::SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>>
    where
        Self: CustomOperation,
    {
        let rv = RVGATE.get_bound(py);
        let param = params.into_pyobject(py)?.cast_into::<PyList>()?;
        Ok(rv
            .call1((
                param.get_item(0)?,
                param.get_item(1)?,
                param.get_item(2)?,
                self.basis().as_str(),
                label,
            ))?
            .unbind())
    }
}

#[pyclass(name="InnerRVGate", extends=PyCustomOperation, module="qiskit.circuit.library.generalized_gates.rv")]
pub struct PyRVGate;

#[pymethods]
impl PyRVGate {
    #[new]
    #[pyo3(signature = (basis = "U"))]
    fn new(basis: &str) -> PyResult<PyClassInitializer<PyRVGate>> {
        let Ok(basis) = EulerBasis::from_str(basis) else {
            return Err(QiskitError::new_err(format!(
                "OneQubitEulerDecomposer: unsupported basis {}",
                basis
            )));
        };

        Ok(PyClassInitializer::new(
            Self,
            PyCustomOperation::new(Box::new(RVGate::new(basis))).into(),
        ))
    }
}
