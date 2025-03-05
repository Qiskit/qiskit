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

use ndarray::ArrayView2;
use num_complex::{Complex64, ComplexFloat};
use qiskit_circuit::{operations::StandardGate, Qubit};

use crate::unitary_compose;

/// For a (controlled) rotation or phase gate, return a tuple ``(Tr(gate) / dim, dim)``.
/// Returns ``None`` if the rotation gate (specified by name) is not supported.
pub fn rotation_trace_and_dim(rotation: StandardGate, angle: f64) -> Option<(Complex64, f64)> {
    let dim = match rotation {
        StandardGate::RXGate
        | StandardGate::RYGate
        | StandardGate::RZGate
        | StandardGate::PhaseGate
        | StandardGate::U1Gate => 2.,
        _ => 4.,
    };

    let trace_over_dim = match rotation {
        StandardGate::RXGate
        | StandardGate::RYGate
        | StandardGate::RZGate
        | StandardGate::RXXGate
        | StandardGate::RYYGate
        | StandardGate::RZZGate
        | StandardGate::RZXGate => Complex64::new((angle / 2.).cos(), 0.),
        StandardGate::CRXGate | StandardGate::CRYGate | StandardGate::CRZGate => {
            Complex64::new(0.5 + 0.5 * (angle / 2.).cos(), 0.)
        }
        StandardGate::PhaseGate | StandardGate::U1Gate => {
            (1. + Complex64::new(0., angle).exp()) / 2.
        }
        StandardGate::CPhaseGate => (3. + Complex64::new(0., angle).exp()) / 4.,
        _ => return None,
    };
    Some((trace_over_dim, dim))
}

pub fn gate_fidelity(
    left: &ArrayView2<Complex64>,
    right: &ArrayView2<Complex64>,
    qargs: Option<&[Qubit]>,
) -> (f64, f64) {
    let dim = left.nrows();

    let left = left.t().mapv(|el| el.conj());
    let product = match dim {
        2 => unitary_compose::matmul_1q(&left.view(), right),
        4 => {
            unitary_compose::matmul_2q(&left.view(), right, qargs.unwrap_or(&[Qubit(0), Qubit(1)]))
        }
        _ => left.dot(right),
    };
    let trace = product.diag().sum();

    let dim = dim as f64;
    let normalized_trace = trace / dim;
    let phase = normalized_trace.arg(); // compute phase difference

    let process_fidelity = normalized_trace.abs().powi(2);
    let gate_fidelity = (dim * process_fidelity + 1.) / (dim + 1.);
    (gate_fidelity, phase)
}
