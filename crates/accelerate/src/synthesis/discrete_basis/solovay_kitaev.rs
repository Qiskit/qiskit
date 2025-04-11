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

use nalgebra::Matrix2;
use numpy::{Complex64, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};

use crate::synthesis::discrete_basis::basic_approximations::{BasicApproximations, GateSequence};

use super::basic_approximations::DiscreteBasisError;
use super::math::{self, group_commutator_decomposition};

/// A stateful implementation of Solovay Kitaev.
///
/// This generates the basic approximation set once as R-tree and re-uses it for
/// each queried decomposition.
#[pyclass]
pub struct SolovayKitaevSynthesis {
    /// The set of basic approximations.
    basic_approximations: BasicApproximations,
    /// Whether to perform runtime checks on the handled matrices/data.
    do_checks: bool,
}

impl SolovayKitaevSynthesis {
    /// Initialize a new instance by generating a set of basic approximations, given a list
    /// of discrete standard gates and a depth. Returns an error if any gate is not discrete.
    pub fn new(
        basis_gates: &[StandardGate],
        depth: usize,
        tol: Option<f64>,
        do_checks: bool,
    ) -> Result<Self, DiscreteBasisError> {
        let basic_approximations = BasicApproximations::generate_from(basis_gates, depth, tol)?;
        Ok(Self {
            basic_approximations,
            do_checks,
        })
    }

    /// Run the Solovay Kitaev algorithm, given the initial unitary as U(2) matrix.
    ///
    /// This matrix is given using [Complex64] numbers, which can limit the precision of the
    /// algorithm. It is preferred to run the algorithm using [synthesize_std], which let's Qiskit
    /// attempt to construct the underlying SO(3) matrix representation using [BigFloat] at
    /// quadruple precision, instead of inferring it from the double precision U(2) matrix.
    pub fn synthesize_matrix(
        &self,
        matrix_u2: &Matrix2<Complex64>,
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let input_sequence = GateSequence::from_u2(matrix_u2, self.do_checks);
        self.synthesize_sequence(&input_sequence, recursion_degree)
    }

    /// Run the Solovay Kitaev algorithm on a standard gate + gate parameters.
    ///
    /// This attempts to directly construct the SO(3) matrix representation to minimize roundoff
    /// errors. If unsuccessful, this falls back onto constructing the standard U(2) matrix
    /// representation and converting it to SO(3).
    pub fn synthesize_std(
        &self,
        gate: &StandardGate,
        params: &[Param],
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let input_sequence = GateSequence::from_std(gate, params)?;
        self.synthesize_sequence(&input_sequence, recursion_degree)
    }

    /// Run Solovay Kitaev given a [GateSequence].
    ///
    /// Mainly an internal helper for other runner methods.
    fn synthesize_sequence(
        &self,
        sequence: &GateSequence,
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        // Run SK recursion.
        let mut output = self.recurse(sequence, recursion_degree);

        // Do minor optimizations on the output sequence, e.g. cancel gate-inverse-pairs, which
        // can arise due to the recursions (even though they should not be present in the initial
        // set of basic approximations).
        output.inverse_cancellation();

        // Compute the error in the SO(3) representation to return with the circuit.
        let circuit = output.to_circuit(Some(sequence))?;
        Ok(circuit)
    }

    /// Run a recursion step for a gate sequence, given a recursion degree.
    ///
    /// If the degree is 0 (recursion root), return the closest approximation in the set of
    /// basic approximations. Otherwise, decompose the difference between the approximation
    /// and the target unitary as balanced group commutator, and recurse on each element.
    fn recurse(&self, sequence: &GateSequence, degree: usize) -> GateSequence {
        // Recursion root: return the best approximation in the precomputed set.
        if degree == 0 {
            if self.do_checks {
                math::assert_so3("Recursion root", &sequence.matrix_so3);
            }
            let basic_approximation = self
                .basic_approximations
                .query(sequence)
                .expect("No basic approximation in root found")
                .clone();
            return basic_approximation;
        }

        // Find a basic approximation of the target sequence...
        let u_n1 = self.recurse(sequence, degree - 1);

        // ... and then improve the delta in between that approximation and the target.
        let delta = sequence.matrix_so3 * u_n1.matrix_so3.transpose();
        let (matrix_vn, matrix_wn) = group_commutator_decomposition(&delta, self.do_checks);
        let vn = GateSequence::from_so3(&matrix_vn, self.do_checks);
        let wn = GateSequence::from_so3(&matrix_wn, self.do_checks);

        // Recurse on the group commutator elements.
        let v_n1 = self.recurse(&vn, degree - 1);
        let w_n1 = self.recurse(&wn, degree - 1);

        v_n1.dot(&w_n1)
            .dot(&v_n1.adjoint())
            .dot(&w_n1.adjoint())
            .dot(&u_n1)
    }
}

#[pymethods]
impl SolovayKitaevSynthesis {
    /// Args:
    ///     basis_gates (list[Gate] | None): A list of discrete (i.e., non-parameterized) standard
    ///         gates. Defaults to ``[H, T, Tdg]``.
    ///     depth (int): The number of basis gate combinations to consider in the basis set. This
    ///         determines how fast (and if) the algorithm converges and should be chosen
    ///         sufficiently high. Defaults to 16.
    ///     tol (float | None): A tolerance determining the granularity of the basic approximations.
    ///         Any sequence whose SO(3) representation is withing :math:`\sqrt{\texttt{tol}}` of
    ///         an existing point, will be discarded. Defaults to ``1e-14``.
    #[new]
    #[pyo3 (signature = (basis_gates=None, depth=16, tol=None, do_checks=false))]
    fn py_new(
        basis_gates: Option<&Bound<PyList>>,
        depth: usize,
        tol: Option<f64>,
        do_checks: bool,
    ) -> PyResult<Self> {
        // Extract list of standard gates from the input. Errors if a gate is not a standard gate.
        let basis_gates: Vec<StandardGate> = match basis_gates {
            None => vec![StandardGate::H, StandardGate::T, StandardGate::Tdg],
            Some(py_gates) => py_gates
                .iter()
                .map(|el| {
                    let py_op = el.extract::<OperationFromPython>()?;
                    match py_op.operation.view() {
                        OperationRef::StandardGate(gate) => Ok(gate),
                        _ => Err(PyValueError::new_err("Only standard gates accepted.")),
                    }
                })
                .collect::<PyResult<_>>()?,
        };

        // Construct self. Errors if a gate is not a discrete standard gate.
        Self::new(&basis_gates, depth, tol, do_checks).map_err(|err| err.into())
    }

    /// Run the Solovay-Kitaev algorithm on a :math:`U(2)` input matrix.
    ///
    /// For better accuracy, it is suggested to use :meth:`synthesize`, which provides the
    /// :class:`.Gate` to decompose and allows Qiskit to internally create a high-accuracy
    /// representation.
    ///
    /// Args:
    ///     matrix (np.ndarray): A 2x2 complex matrix representing a 1-qubit gate.
    ///     recursion_degree (int): The recursion degree of the algorithm.
    ///
    /// Returns:
    ///     CircuitData: The ``CircuitData`` implementing the approximation.
    #[pyo3(name = "synthesize_matrix")]
    #[pyo3(signature = (gate_matrix, recursion_degree))]
    fn py_synthesize_matrix(
        &self,
        gate_matrix: PyReadonlyArray2<Complex64>,
        recursion_degree: usize,
    ) -> PyResult<CircuitData> {
        let view = Matrix2::new(
            *gate_matrix.get((0, 0)).unwrap(),
            *gate_matrix.get((0, 1)).unwrap(),
            *gate_matrix.get((1, 0)).unwrap(),
            *gate_matrix.get((1, 1)).unwrap(),
        );
        self.synthesize_matrix(&view, recursion_degree)
            .map_err(|err| err.into())
    }

    /// Run the Solovay-Kitaev algorithm on a standard gate.
    ///
    /// Args:
    ///     gate (Gate): The standard gate to approximate.
    ///     recursion_degree (int): The recursion degree of the algorithm.
    ///
    /// Returns:
    ///     CircuitData: The ``CircuitData`` implementing the approximation.
    fn synthesize(
        &self,
        gate: OperationFromPython,
        recursion_degree: usize,
    ) -> PyResult<CircuitData> {
        let params = gate.params;
        let gate = match gate.operation.view() {
            OperationRef::StandardGate(gate) => gate,
            _ => return Err(PyValueError::new_err("Only standard gates are supported.")),
        };
        self.synthesize_std(&gate, &params, recursion_degree)
            .map_err(|err| err.into())
    }

    /// Query the basic approximation for a :class:`.Gate`.
    ///
    /// Args:
    ///     sequence (Gate): The gate sequence to find the approximation of.
    ///
    /// Returns:
    ///     CircuitData: The sequence in the set of basic approximations closest to the input.
    fn find_basic_approximation(&self, gate: OperationFromPython) -> PyResult<CircuitData> {
        let params = gate.params;
        let sequence = match gate.operation.view() {
            OperationRef::StandardGate(std_gate) => {
                match GateSequence::from_std(&std_gate, &params) {
                    Ok(sequence) => sequence,
                    Err(e) => return Err(PyValueError::new_err(e.to_string())),
                }
            }
            _ => {
                let matrix_u2 = match gate.operation.matrix(&params) {
                    Some(matrix) => Matrix2::new(
                        matrix[(0, 0)],
                        matrix[(0, 1)],
                        matrix[(1, 0)],
                        matrix[(1, 1)],
                    ),
                    None => {
                        return Err(PyValueError::new_err(
                            "Failed to construct matrix representation.",
                        ))
                    }
                };
                GateSequence::from_u2(&matrix_u2, self.do_checks)
            }
        };

        let approximation = self
            .basic_approximations
            .query(&sequence)
            .expect("No basic approximation found");

        approximation
            .to_circuit(Some(&sequence))
            .map_err(|e| e.into())
    }
}
