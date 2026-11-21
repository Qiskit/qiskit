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

use num_complex::Complex64;
use num_complex::ComplexFloat;
use std::f64::consts::PI;

use crate::gate_metrics::rotation_trace_and_dim;
use qiskit_circuit::operations::StandardGate;

// Default value for tolerance level. It is used to check the fidely tolerance.
pub const MINIMUM_TOL: f64 = 1e-12;

/// Fidelity-based computation to check whether an operation `G` is equivalent
/// to identity up to a global phase.
///
/// # Arguments
///
/// * `tr_over_dim`: `|Tr(G)| / dim(G)`.
/// * `dim`: `dim(G)`.
/// * `tol`: tolerance.
///
/// # Returns
///
/// * `Some(update to the global phase)` if the operation can be removed.
/// * `None` if the operation cannot be removed.
pub fn average_gate_fidelity_below_tol(tr_over_dim: Complex64, dim: f64, tol: f64) -> Option<f64> {
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    if (1. - gate_fidelity).abs() < tol {
        Some(tr_over_dim.arg())
    } else {
        None
    }
}

/// For a given angle, if it is a multiple of PI/k, calculate the multiple mod (4*k),
/// Otherwise, return `None`.
/// E.g, if the angle is a multiple m of PI/4 then it returns m, where 0 <= m < 16,
/// and if the angle is a multiple m of PI/2 then it returns m, where 0 <= m < 8.
/// Panics if the StandardGate is not a rotation gate.
pub fn is_angle_close_to_multiple_of_pi_k(
    gate: StandardGate,
    k: usize,
    angle: f64,
    tol: f64,
) -> Option<usize> {
    let closest_ratio = angle * (k as f64) / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / (k as f64);
    let theta = angle - closest_angle;

    // Trace and dimension calculation of rotation matrices
    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, theta)
        .expect("Since only supported rotation gates are given, the result is not None");

    // fidelity-based tolerance
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    let rem = 4 * k as i64;
    if (1. - gate_fidelity).abs() < tol {
        Some((closest_integer as i64).rem_euclid(rem) as usize)
    } else {
        None
    }
}

/// Check whether two sparse Paulis commute.
///
/// Each Pauli is given in the sparse ZX format, that is by its Z and X components and the
/// indices of the qubits it acts on.
///
/// The qubit indices of each Pauli must be sorted. This allows to check commutation in
/// linear time, by computing the parity of anti-commuting terms on the common qubits.
pub fn sparse_paulis_commute<T: Ord>(
    z1: &[bool],
    x1: &[bool],
    qargs1: &[T],
    z2: &[bool],
    x2: &[bool],
    qargs2: &[T],
) -> bool {
    let mut parity = false;
    let (n1, n2) = (qargs1.len(), qargs2.len());
    let (mut i1, mut i2) = (0, 0);
    while i1 < n1 && i2 < n2 {
        match qargs1[i1].cmp(&qargs2[i2]) {
            std::cmp::Ordering::Less => i1 += 1,
            std::cmp::Ordering::Greater => i2 += 1,
            std::cmp::Ordering::Equal => {
                parity ^= (x1[i1] && z2[i2]) ^ (z1[i1] && x2[i2]);
                i1 += 1;
                i2 += 1;
            }
        }
    }
    !parity
}

#[cfg(test)]
mod tests {
    use super::sparse_paulis_commute;

    // Sparse ZX format of single-qubit Paulis: (z, x).
    const X: (bool, bool) = (false, true);
    const Y: (bool, bool) = (true, true);
    const Z: (bool, bool) = (true, false);

    /// Check commutation of two sparse Paulis, each given as a list of `(pauli, qubit)` terms.
    fn commute(p1: &[((bool, bool), u32)], p2: &[((bool, bool), u32)]) -> bool {
        let split = |p: &[((bool, bool), u32)]| {
            let z: Vec<bool> = p.iter().map(|((z, _), _)| *z).collect();
            let x: Vec<bool> = p.iter().map(|((_, x), _)| *x).collect();
            let q: Vec<u32> = p.iter().map(|(_, q)| *q).collect();
            (z, x, q)
        };
        let (z1, x1, q1) = split(p1);
        let (z2, x2, q2) = split(p2);
        let result = sparse_paulis_commute(&z1, &x1, &q1, &z2, &x2, &q2);
        // Commutation is symmetric.
        assert_eq!(result, sparse_paulis_commute(&z2, &x2, &q2, &z1, &x1, &q1));
        result
    }

    #[test]
    fn test_single_qubit_paulis() {
        assert!(commute(&[(X, 0)], &[(X, 0)]));
        assert!(commute(&[(Y, 0)], &[(Y, 0)]));
        assert!(!commute(&[(X, 0)], &[(Z, 0)]));
        assert!(!commute(&[(X, 0)], &[(Y, 0)]));
        assert!(!commute(&[(Y, 0)], &[(Z, 0)]));
    }

    #[test]
    fn test_disjoint_qubits() {
        assert!(commute(&[(X, 0)], &[(Z, 1)]));
        assert!(commute(&[(X, 0), (Y, 2)], &[(Z, 1), (Z, 3)]));
        assert!(commute(&[], &[(Z, 1)]));
        assert!(commute(&[], &[]));
    }

    #[test]
    fn test_multi_qubit_paulis() {
        // Two anti-commuting positions: X0 X1 and Z0 Z1 commute.
        assert!(commute(&[(X, 0), (X, 1)], &[(Z, 0), (Z, 1)]));
        // One anti-commuting position: X0 Z1 and Z0 Z1 anti-commute.
        assert!(!commute(&[(X, 0), (Z, 1)], &[(Z, 0), (Z, 1)]));
        // Partial overlap on qubits 2 and 5: Y2 X5 and Z2 Z5 commute (two anti-commuting
        // positions), and the non-overlapping qubits do not matter.
        assert!(commute(
            &[(X, 0), (Y, 2), (X, 5)],
            &[(Z, 1), (Z, 2), (Z, 3), (Z, 5)]
        ));
        // Partial overlap on qubit 2 only: Y2 and Z2 anti-commute.
        assert!(!commute(
            &[(X, 0), (Y, 2), (X, 6)],
            &[(Z, 1), (Z, 2), (Z, 5)]
        ));
    }
}
