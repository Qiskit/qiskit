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

// AI Attribution:
// This module was developed with assistance from GitHub Copilot integrated in VS Code. The underlying model : Claude Haiku 4.5.
// Portions of the Pauli generator mapping logic and SoA layout were generated
// and then manually verified for mathematical correctness against the commutation logic.

// This module maps standard quantum gates to their Hamiltonian generators H_gate such that:
//
//   gate = exp(-i * H_gate)   (up to global phase)
//
// The generator is returned as a SparseObservable (a sum of Pauli tensor products).
//
// For single-qubit gates: X = exp(-i*(pi/2)*X), H = exp(-i*(pi/2)*(X+Z)/sqrt(2)), etc.
// For multi-qubit controlled gates: CX = exp(-i*(pi/4)*(ZX - ZI - IX)), etc.
// For rotation gates: RX(t) = exp(-i*(t/2)*X), RY(t) = exp(-i*(t/2)*Y), etc.
// The SoA (Struct-of-Arrays) layout is used: `bit_terms`, `indices`, and `boundaries` are
// flattened arrays. Term `i` uses bit_terms[boundaries[i]..boundaries[i+1]] and
// indices[boundaries[i]..boundaries[i+1]] for its Pauli operators and qubit targets.

use super::BitTerm;
use super::SparseObservable;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_util::util::{C_ECR_FACTOR, C_FRAC_PI_2, C_FRAC_PI_4, C_FRAC_PI_8, C_ZERO, c64};

const C_M_FRAC_PI_4: Complex64 = c64(-C_FRAC_PI_4.re, 0.0);
const C_M_FRAC_PI_8: Complex64 = c64(-C_FRAC_PI_8.re, 0.0);
const C_M_ECR_FACTOR: Complex64 = c64(-C_ECR_FACTOR.re, 0.0);

const BETA_TOLERANCE: f64 = 1e-10;

/// For parametric gates (e.g., `RX(theta)`), the generator $H$ depends on the
/// gate parameters (e.g., $H = (\theta/2)X$). This function extracts parameter values
/// from the `params` slice to compute the concrete coefficients for the returned
/// `SparseObservable`.
///
/// If parameters are missing or symbolic (non-`Float`), it defaults to a coefficient
/// of 1.0. This allows the commutation checker to still prove commutation in cases
/// where the generator's Pauli structure alone is sufficient (e.g., `[theta*X, X] = 0`
/// for any `theta`), but it avoids attempting to store parametric expressions, which
/// `SparseObservable` does not currently support.
pub fn generator_observable(gate: StandardGate, params: &[Param]) -> Option<SparseObservable> {
    let _params = params;
    let num_qubits = gate.num_qubits();

    use BitTerm::*;

    // Global phase gate
    if num_qubits == 0 {
        if let StandardGate::GlobalPhase = gate {
            // Global Phase is exp(i * theta).
            // Generator H such that exp(-i * H) = exp(i * theta) -> H = -theta.
            let mut theta = 1.0;
            if let [Param::Float(t)] = _params {
                theta = *t;
            }
            return Some(
                SparseObservable::new(
                    0,
                    vec![c64(-theta, 0.0)],
                    vec![],     // no paulis -> Identity
                    vec![],     // no target qubits
                    vec![0, 0], // 1 term of length 0
                )
                .expect("invalid 0-qubit generator layout"),
            );
        }
        return None;
    }

    // Single-qubit gates
    if num_qubits == 1 {
        let (coeffs, terms, indices) = match gate {
            // H = exp(-i*(pi/2)*(X + Z)/sqrt(2))
            // => H_gen = (pi/2)*(X + Z)/sqrt(2) = (pi / (2*sqrt(2))) * X + (pi / (2*sqrt(2))) * Z
            // Numerically: pi / (2*sqrt(2)) ≈ 1.1107...
            // Note: the sign must be negative so H_gate uses coefficients +1/sqrt(2) each.
            StandardGate::H => {
                let c = C_ECR_FACTOR.re;
                (vec![c64(c, 0.0), c64(c, 0.0)], vec![X, Z], vec![0, 0])
            }
            // X = exp(-i*(pi/2)*X), Y = exp(-i*(pi/2)*Y), Z = exp(-i*(pi/2)*Z)
            StandardGate::X => (vec![C_FRAC_PI_2], vec![X], vec![0]),
            StandardGate::Y => (vec![C_FRAC_PI_2], vec![Y], vec![0]),
            StandardGate::Z => (vec![C_FRAC_PI_2], vec![Z], vec![0]),
            // Identity: exp(-i * 0) = I.
            StandardGate::I => (vec![C_ZERO], vec![], vec![]),
            // S = exp(-i*(pi/4)*Z), Sdg = exp(-i*(-pi/4)*Z)
            StandardGate::S => (vec![C_FRAC_PI_4], vec![Z], vec![0]),
            StandardGate::Sdg => (vec![C_M_FRAC_PI_4], vec![Z], vec![0]),
            // T = exp(-i*(pi/8)*Z), Tdg = exp(-i*(-pi/8)*Z)
            StandardGate::T => (vec![C_FRAC_PI_8], vec![Z], vec![0]),
            StandardGate::Tdg => (vec![C_M_FRAC_PI_8], vec![Z], vec![0]),
            // SX = exp(-i*(pi/4)*X), SXdg = exp(-i*(-pi/4)*X)
            StandardGate::SX => (vec![C_FRAC_PI_4], vec![X], vec![0]),
            StandardGate::SXdg => (vec![C_M_FRAC_PI_4], vec![X], vec![0]),
            // RX(t) = exp(-i*(t/2)*X), RY(t) = exp(-i*(t/2)*Y)
            // RZ(t) = exp(-i*(t/2)*Z), Phase(t) = exp(-i*(t/2)*Z) (same generator)
            StandardGate::RX | StandardGate::RY | StandardGate::RZ | StandardGate::Phase => {
                // Qiskit's `_generator_observable` falls back to `1.0` if no parameters are available or the parameter is an unbound expression.
                // This corresponds effectively to returning the base operator for the Pauli (e.g. `X.generator() == X`).
                // In normal workflows the `params` tuple is fully concrete during commutation logic (i.e., `Float`).
                let theta = if let [Param::Float(t)] = _params {
                    *t
                } else {
                    1.0
                };
                let term = match gate {
                    StandardGate::RX => X,
                    StandardGate::RY => Y,
                    _ => Z,
                };
                (vec![c64(theta / 2.0, 0.0)], vec![term], vec![0])
            }
            _ => return None,
        };

        // For 1-qubit gates each term has exactly 1 Pauli operator, so
        // boundaries = [0, 1, 2, ..., N].
        // Exception: Identity gate has 1 term of length 0.
        let boundaries: Vec<usize> = if gate == StandardGate::I {
            vec![0, 0]
        } else {
            (0..=terms.len()).collect()
        };

        return Some(
            SparseObservable::new(num_qubits, coeffs, terms, indices, boundaries)
                .expect("invalid 1-qubit generator layout"),
        );
    }

    // Multi-qubit gates
    // Returns (coeffs, bit_terms, indices, boundaries) in SoA layout.
    // Qubit ordering convention: index 0 = qubit 0 = LEAST significant (rightmost in Qiskit strings).
    // For a 2q gate: control = qubit 0, target = qubit 1.
    let (coeffs, bit_terms, indices, boundaries) = match gate {
        // CX (CNOT): CX = exp(-i*(pi/4)*(Z0*X1 - Z0 - X1))
        // Generator H = (pi/4)*(Z0*X1 - Z0 - X1)
        // Terms: [Z0X1 coeff=+pi/4], [Z0 coeff=-pi/4], [X1 coeff=-pi/4]
        StandardGate::CX => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            // Term 0: Z(q0) X(q1);  Term 1: Z(q0);  Term 2: X(q1)
            vec![Z, X, Z, X],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CY: CY = exp(-i*(pi/4)*(Z0*Y1 - Z0 - Y1))
        StandardGate::CY => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![Z, Y, Z, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CZ: CZ = exp(-i*(pi/4)*(Z0*Z1 - Z0 - Z1))
        StandardGate::CZ => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CS (sqrt(CZ)): CS = exp(-i*(pi/8)*(Z0 + Z1 - Z0*Z1))
        // Generator H = (pi/8)*(Z0 + Z1 - Z0*Z1)
        StandardGate::CS => (
            vec![C_FRAC_PI_8, C_FRAC_PI_8, C_M_FRAC_PI_8],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CSdg (inv sqrt(CZ)): factor +pi/8
        StandardGate::CSdg => (
            vec![C_M_FRAC_PI_8, C_M_FRAC_PI_8, C_FRAC_PI_8],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CSX (sqrt(CX)/controlled-SX): factor -pi/8
        StandardGate::CSX => (
            vec![C_FRAC_PI_8, C_FRAC_PI_8, C_M_FRAC_PI_8],
            vec![Z, X, Z, X],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CRX(t): CRX(t) = exp(-i*(t/4)*(-Z0*X1 + X1))
        // Generator = -t/4 * Z0*X1 + t/4 * X1
        StandardGate::CRX => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![c64(-t / 4.0, 0.0), c64(t / 4.0, 0.0)],
                vec![Z, X, X],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CRY(t): Generator = -t/4 * Z0*Y1 + t/4 * Y1
        StandardGate::CRY => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![c64(-t / 4.0, 0.0), c64(t / 4.0, 0.0)],
                vec![Z, Y, Y],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CRZ(t): Generator = -t/4 * Z0*Z1 + t/4 * Z1
        StandardGate::CRZ => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![c64(-t / 4.0, 0.0), c64(t / 4.0, 0.0)],
                vec![Z, Z, Z],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CPhase(t): Generator = -t/4 * Z0*Z1 + t/4 * Z0 + t/4 * Z1
        // (same as CRZ but shifts both Z0 and Z1, not just Z1)
        StandardGate::CPhase => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![c64(-t / 4.0, 0.0), c64(t / 4.0, 0.0), c64(t / 4.0, 0.0)],
                vec![Z, Z, Z, Z],
                vec![0, 1, 0, 1],
                vec![0, 2, 3, 4],
            )
        }
        // Swap: Swap = exp(-i*(pi/4)*(X0*X1 + Y0*Y1 + Z0*Z1))
        // (note: Swap = exp(-i*pi/4*(XX+YY+ZZ)) treats Swap as "swap up to phase for each sector")
        StandardGate::Swap => (
            vec![C_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![X, X, Y, Y, Z, Z],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 2, 4, 6],
        ),
        // ISwap: ISwap = exp(-i*(-pi/4)*(X0*X1 + Y0*Y1))
        StandardGate::ISwap => (
            vec![C_M_FRAC_PI_4, C_M_FRAC_PI_4],
            vec![X, X, Y, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        ),
        // RXX(t) = exp(-i*(t/2)*X0*X1)
        StandardGate::RXX => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (vec![c64(t / 2.0, 0.0)], vec![X, X], vec![0, 1], vec![0, 2])
        }
        // RYY(t) = exp(-i*(t/2)*Y0*Y1)
        StandardGate::RYY => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (vec![c64(t / 2.0, 0.0)], vec![Y, Y], vec![0, 1], vec![0, 2])
        }
        // RZZ(t) = exp(-i*(t/2)*Z0*Z1)
        StandardGate::RZZ => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (vec![c64(t / 2.0, 0.0)], vec![Z, Z], vec![0, 1], vec![0, 2])
        }
        // RZX(t) = exp(-i*(t/2)*Z0*X1)
        StandardGate::RZX => {
            let t = if let [Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (vec![c64(t / 2.0, 0.0)], vec![Z, X], vec![0, 1], vec![0, 2])
        }
        // XXPlusYY(theta, beta): Generator = (theta/4)*(X0*X1 + Y0*Y1)
        // (the beta angle just rotates the YY axis; for the commutation check only XX+YY matters)
        StandardGate::XXPlusYY | StandardGate::XXMinusYY => {
            let t = if let [Param::Float(theta), ..] = _params {
                *theta
            } else {
                1.0
            };

            // The beta angle rotates the YY axis. Ensure beta=0 (or assert it) so commutation
            // is strictly XX +/- YY. Otherwise, fallback to matrix checking.
            let beta = if let [_, Param::Float(b)] = _params {
                *b
            } else {
                return None;
            };

            if beta.abs() > BETA_TOLERANCE {
                return None;
            }

            match gate {
                StandardGate::XXPlusYY => (
                    vec![c64(t / 4.0, 0.0), c64(t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                StandardGate::XXMinusYY => (
                    vec![c64(t / 4.0, 0.0), c64(-t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                _ => unreachable!(),
            }
        }
        // CCX (Toffoli): CCX = exp(-i*(pi/8)*(Z0*Z1*X2 - Z0*X2 - Z1*X2 - Z0*Z1 + Z0 + Z1 + X2))
        // 7 terms in total.
        // qubit ordering: q0=ctrl0, q1=ctrl1, q2=target
        StandardGate::CCX => (
            vec![
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
            ],
            vec![Z, Z, X, Z, X, Z, X, Z, Z, Z, Z, X],
            vec![0, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2],
            vec![0, 3, 5, 7, 9, 10, 11, 12],
        ),
        // CCZ: CCZ = exp(-i*(pi/8)*(Z0*Z1*Z2 - Z0*Z2 - Z1*Z2 - Z0*Z1 + Z0 + Z1 + Z2))
        StandardGate::CCZ => (
            vec![
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
            ],
            vec![Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z],
            vec![0, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2],
            vec![0, 3, 5, 7, 9, 10, 11, 12],
        ),
        // CSwap (Fredkin): CSwap = exp(-i*(pi/8)*(Z0 - Z0*X1*X2 - Z0*Y1*Y2 - Z0*Z1*Z2 + X1*X2 + Y1*Y2 + Z1*Z2))
        // 7 terms: Z0(-ZXX=ZYY=ZZZ), +XX, +YY, +ZZ
        StandardGate::CSwap => (
            vec![
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
            ],
            vec![Z, Z, X, X, Z, Y, Y, Z, Z, Z, X, X, Y, Y, Z, Z],
            vec![0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2],
            vec![0, 1, 4, 7, 10, 12, 14, 16],
        ),
        // ECR: ECR = exp(-i * (pi/2/sqrt(2)) * (IX - XY))
        // Terms: X1 with coeff +pi/(2*sqrt(2)), X1Y0 with coeff -pi/(2*sqrt(2))
        StandardGate::ECR => (
            vec![C_ECR_FACTOR, C_M_ECR_FACTOR],
            vec![X, Y, X],
            vec![0, 0, 1],
            vec![0, 1, 3],
        ),
        _ => return None,
    };

    Some(
        SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
            .expect("invalid multi-qubit generator layout"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::RX, &[]).expect("Rx should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }

    #[test]
    fn cx_has_some_generator() {
        let obs = generator_observable(StandardGate::CX, &[]).expect("CX should have a generator");
        assert_eq!(obs.num_terms(), 3);
    }

    #[test]
    fn ccz_has_seven_terms() {
        let obs =
            generator_observable(StandardGate::CCZ, &[]).expect("CCZ should have a generator");
        assert_eq!(obs.num_terms(), 7);
    }

    #[test]
    fn cswap_has_seven_terms() {
        let obs =
            generator_observable(StandardGate::CSwap, &[]).expect("CSwap should have a generator");
        assert_eq!(obs.num_terms(), 7);
    }

    #[test]
    fn ecr_has_two_terms() {
        let obs =
            generator_observable(StandardGate::ECR, &[]).expect("ECR should have a generator");
        assert_eq!(obs.num_terms(), 2);
    }
}

