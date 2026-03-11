// src/sparse_observable/standard_generators.rs
//
// This module maps standard quantum gates to their Hamiltonian generators H_gate such that:
//
//   gate = exp(-i * H_gate)   (up to global phase)
//
// The generator is returned as a SparseObservable (a sum of Pauli tensor products).
//
// For single-qubit gates: X = exp(-i*(pi/2)*X), H = exp(-i*(pi/2)*(X+Z)/sqrt(2)), etc.
// For multi-qubit controlled gates: CX = exp(-i*(pi/4)*(ZX - Z - X)), etc.
// For rotation gates: RX(t) = exp(-i*(t/2)*X), RY(t) = exp(-i*(t/2)*Y), etc.
//
// The SoA (Struct-of-Arrays) layout is used: `bit_terms`, `indices`, and `boundaries` are
// flattened arrays. Term `i` uses bit_terms[boundaries[i]..boundaries[i+1]] and
// indices[boundaries[i]..boundaries[i+1]] for its Pauli operators and qubit targets.

use super::BitTerm;
use super::SparseObservable;
use num_complex::Complex64;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::StandardGate;

/// Return a `SparseObservable` H such that `gate ≈ exp(-i * H)` (up to global phase),
/// or `None` if no generator is known for this gate.
pub fn generator_observable(
    gate: StandardGate,
    params: &[qiskit_circuit::operations::Param],
) -> Option<SparseObservable> {
    let _params = params;
    let num_qubits = gate.num_qubits();

    use BitTerm::*;

    // ─── Single-qubit gates ───────────────────────────────────────────────────
    if num_qubits == 1 {
        let (coeffs, terms, indices) = match gate {
            // H = exp(-i*(pi/2)*(X + Z)/sqrt(2))
            // => H_gen = (pi/2)*(X + Z)/sqrt(2) = (pi / (2*sqrt(2))) * X + (pi / (2*sqrt(2))) * Z
            // Numerically: pi / (2*sqrt(2)) ≈ 1.1107...
            // Note: the sign must be negative so H_gate uses coefficients +1/sqrt(2) each.
            StandardGate::H => {
                let c = std::f64::consts::PI / (2.0 * std::f64::consts::SQRT_2);
                (
                    vec![Complex64::new(c, 0.0), Complex64::new(c, 0.0)],
                    vec![X, Z],
                    vec![0u32, 0u32],
                )
            }
            // X = exp(-i*(pi/2)*X), Y = exp(-i*(pi/2)*Y), Z = exp(-i*(pi/2)*Z)
            StandardGate::X => (
                vec![Complex64::new(std::f64::consts::PI / 2.0, 0.0)],
                vec![X],
                vec![0u32],
            ),
            StandardGate::Y => (
                vec![Complex64::new(std::f64::consts::PI / 2.0, 0.0)],
                vec![Y],
                vec![0u32],
            ),
            StandardGate::Z => (
                vec![Complex64::new(std::f64::consts::PI / 2.0, 0.0)],
                vec![Z],
                vec![0u32],
            ),
            // S = exp(-i*(pi/4)*Z), Sdg = exp(-i*(-pi/4)*Z)
            StandardGate::S => (
                vec![Complex64::new(std::f64::consts::PI / 4.0, 0.0)],
                vec![Z],
                vec![0u32],
            ),
            StandardGate::Sdg => (
                vec![Complex64::new(-std::f64::consts::PI / 4.0, 0.0)],
                vec![Z],
                vec![0u32],
            ),
            // T = exp(-i*(pi/8)*Z), Tdg = exp(-i*(-pi/8)*Z)
            StandardGate::T => (
                vec![Complex64::new(std::f64::consts::PI / 8.0, 0.0)],
                vec![Z],
                vec![0u32],
            ),
            StandardGate::Tdg => (
                vec![Complex64::new(-std::f64::consts::PI / 8.0, 0.0)],
                vec![Z],
                vec![0u32],
            ),
            // SX = exp(-i*(pi/4)*X), SXdg = exp(-i*(-pi/4)*X)
            StandardGate::SX => (
                vec![Complex64::new(std::f64::consts::PI / 4.0, 0.0)],
                vec![X],
                vec![0u32],
            ),
            StandardGate::SXdg => (
                vec![Complex64::new(-std::f64::consts::PI / 4.0, 0.0)],
                vec![X],
                vec![0u32],
            ),
            // RX(t) = exp(-i*(t/2)*X), RY(t) = exp(-i*(t/2)*Y)
            // RZ(t) = exp(-i*(t/2)*Z), Phase(t) = exp(-i*(t/2)*Z) (same generator)
            StandardGate::RX | StandardGate::RY | StandardGate::RZ | StandardGate::Phase => {
                let theta = if let [qiskit_circuit::operations::Param::Float(t)] = _params {
                    *t
                } else {
                    1.0
                };
                let term = match gate {
                    StandardGate::RX => X,
                    StandardGate::RY => Y,
                    _ => Z,
                };
                (
                    vec![Complex64::new(theta / 2.0, 0.0)],
                    vec![term],
                    vec![0u32],
                )
            }
            _ => return None,
        };

        // For 1-qubit gates each term has exactly 1 Pauli operator, so
        // boundaries = [0, 1, 2, ..., N].
        let boundaries: Vec<usize> = (0..=terms.len()).collect();

        return Some(
            SparseObservable::new(num_qubits, coeffs, terms, indices, boundaries)
                .expect("invalid 1-qubit generator layout"),
        );
    }

    // ─── Multi-qubit gates ────────────────────────────────────────────────────
    // Returns (coeffs, bit_terms, indices, boundaries) in SoA layout.
    // Qubit ordering convention: index 0 = qubit 0 = LEAST significant (rightmost in Qiskit strings).
    // For a 2q gate: control = qubit 0, target = qubit 1.
    let (coeffs, bit_terms, indices, boundaries) = match gate {
        // CX (CNOT): CX = exp(-i*(pi/4)*(Z0*X1 - Z0 - X1))
        // Generator H = (pi/4)*(Z0*X1 - Z0 - X1)
        // Terms: [Z0X1 coeff=+pi/4], [Z0 coeff=-pi/4], [X1 coeff=-pi/4]
        StandardGate::CX => {
            let s = std::f64::consts::PI / 4.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
                // Term 0: Z(q0) X(q1);  Term 1: Z(q0);  Term 2: X(q1)
                vec![Z, X, Z, X],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CY: CY = exp(-i*(pi/4)*(Z0*Y1 - Z0 - Y1))
        StandardGate::CY => {
            let s = std::f64::consts::PI / 4.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
                vec![Z, Y, Z, Y],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CZ: CZ = exp(-i*(pi/4)*(Z0*Z1 - Z0 - Z1))
        StandardGate::CZ => {
            let s = std::f64::consts::PI / 4.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
                vec![Z, Z, Z, Z],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CS (sqrt(CZ)): CS = exp(-i*(pi/8)*(Z0*Z1 - Z0 - Z1))
        StandardGate::CS => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
                vec![Z, Z, Z, Z],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CSdg (inv sqrt(CZ)): CSdg = exp(-i*(-pi/8)*(Z0*Z1 - Z0 - Z1))
        StandardGate::CSdg => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(-s, 0.0),
                    Complex64::new(s, 0.0),
                    Complex64::new(s, 0.0),
                ],
                vec![Z, Z, Z, Z],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CSX (sqrt(CX)/controlled-SX): CSX = exp(-i*(pi/8)*(Z0*X1 - Z0 - X1))
        StandardGate::CSX => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
                vec![Z, X, Z, X],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // CRX(t): CRX(t) = exp(-i*(t/4)*(-Z0*X1 + X1))
        // Generator = -t/4 * Z0*X1 + t/4 * X1
        StandardGate::CRX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, X, X],
                vec![0u32, 1u32, 1u32],
                vec![0usize, 2usize, 3usize],
            )
        }
        // CRY(t): Generator = -t/4 * Z0*Y1 + t/4 * Y1
        StandardGate::CRY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, Y, Y],
                vec![0u32, 1u32, 1u32],
                vec![0usize, 2usize, 3usize],
            )
        }
        // CRZ(t): Generator = -t/4 * Z0*Z1 + t/4 * Z1
        StandardGate::CRZ => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, Z, Z],
                vec![0u32, 1u32, 1u32],
                vec![0usize, 2usize, 3usize],
            )
        }
        // CPhase(t): Generator = -t/4 * Z0*Z1 + t/4 * Z0 + t/4 * Z1
        // (same as CRZ but shifts both Z0 and Z1, not just Z1)
        StandardGate::CPhase => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![
                    Complex64::new(-t / 4.0, 0.0),
                    Complex64::new(t / 4.0, 0.0),
                    Complex64::new(t / 4.0, 0.0),
                ],
                vec![Z, Z, Z, Z],
                vec![0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 3usize, 4usize],
            )
        }
        // ECR = (1/sqrt(2)) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
        // ECR = exp(i*pi/4) * RZX(pi/2) -- no clean Pauli generator like CX
        // Generator: ECR = exp(-i * (-pi/4) * (X0 - X0*Y1))
        // i.e. H_ECR = (-pi/4) * (X0 - X0*Y1) = (-pi/4)*X0 + (pi/4)*X0*Y1
        StandardGate::ECR => (
            vec![
                Complex64::new(-std::f64::consts::PI / 4.0, 0.0),
                Complex64::new(std::f64::consts::PI / 4.0, 0.0),
            ],
            // Term 0: X(q0);  Term 1: X(q0) Y(q1)
            vec![X, X, Y],
            vec![0u32, 0u32, 1u32],
            vec![0usize, 1usize, 3usize],
        ),
        // Swap: Swap = exp(-i*(pi/4)*(X0*X1 + Y0*Y1 + Z0*Z1))
        // (note: Swap = exp(-i*pi/4*(XX+YY+ZZ)) treats Swap as "swap up to phase for each sector")
        StandardGate::Swap => {
            let s = std::f64::consts::PI / 4.0;
            (
                vec![
                    Complex64::new(s, 0.0),
                    Complex64::new(s, 0.0),
                    Complex64::new(s, 0.0),
                ],
                vec![X, X, Y, Y, Z, Z],
                vec![0u32, 1u32, 0u32, 1u32, 0u32, 1u32],
                vec![0usize, 2usize, 4usize, 6usize],
            )
        }
        // ISwap: ISwap = exp(-i*(pi/4)*(X0*X1 + Y0*Y1))  (missing ZZ part vs Swap)
        StandardGate::ISwap => (
            vec![
                Complex64::new(std::f64::consts::PI / 4.0, 0.0),
                Complex64::new(std::f64::consts::PI / 4.0, 0.0),
            ],
            vec![X, X, Y, Y],
            vec![0u32, 1u32, 0u32, 1u32],
            vec![0usize, 2usize, 4usize],
        ),
        // RXX(t) = exp(-i*(t/2)*X0*X1)
        StandardGate::RXX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 2.0, 0.0)],
                vec![X, X],
                vec![0u32, 1u32],
                vec![0usize, 2usize],
            )
        }
        // RYY(t) = exp(-i*(t/2)*Y0*Y1)
        StandardGate::RYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 2.0, 0.0)],
                vec![Y, Y],
                vec![0u32, 1u32],
                vec![0usize, 2usize],
            )
        }
        // RZZ(t) = exp(-i*(t/2)*Z0*Z1)
        StandardGate::RZZ => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 2.0, 0.0)],
                vec![Z, Z],
                vec![0u32, 1u32],
                vec![0usize, 2usize],
            )
        }
        // RZX(t) = exp(-i*(t/2)*Z0*X1)
        StandardGate::RZX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 2.0, 0.0)],
                vec![Z, X],
                vec![0u32, 1u32],
                vec![0usize, 2usize],
            )
        }
        // XXPlusYY(theta, beta): Generator = (theta/4)*(X0*X1 + Y0*Y1)
        // (the beta angle just rotates the YY axis; for the commutation check only XX+YY matters)
        StandardGate::XXPlusYY | StandardGate::XXMinusYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta), ..] = _params {
                *theta
            } else {
                1.0
            };
            match gate {
                StandardGate::XXPlusYY => (
                    vec![Complex64::new(t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0u32, 1u32, 0u32, 1u32],
                    vec![0usize, 2usize, 4usize],
                ),
                StandardGate::XXMinusYY => (
                    vec![Complex64::new(t / 4.0, 0.0), Complex64::new(-t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0u32, 1u32, 0u32, 1u32],
                    vec![0usize, 2usize, 4usize],
                ),
                _ => unreachable!(),
            }
        }
        // CCX (Toffoli): CCX = exp(-i*(pi/8)*(Z0*Z1*X2 - Z0*X2 - Z1*X2 - Z0*Z1 + Z0 + Z1 + X2))
        // 7 terms in total.
        // qubit ordering: q0=ctrl0, q1=ctrl1, q2=target
        StandardGate::CCX => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(s, 0.0),  // Z0 Z1 X2
                    Complex64::new(-s, 0.0), // Z0 X2
                    Complex64::new(-s, 0.0), // Z1 X2
                    Complex64::new(-s, 0.0), // Z0 Z1
                    Complex64::new(s, 0.0),  // Z0
                    Complex64::new(s, 0.0),  // Z1
                    Complex64::new(s, 0.0),  // X2
                ],
                // Term 0: Z(q0) Z(q1) X(q2)  — 3 items
                // Term 1: Z(q0) X(q2)          — 2 items
                // Term 2: Z(q1) X(q2)          — 2 items
                // Term 3: Z(q0) Z(q1)          — 2 items
                // Term 4: Z(q0)                 — 1 item
                // Term 5: Z(q1)                 — 1 item
                // Term 6: X(q2)                 — 1 item
                // Total: 3+2+2+2+1+1+1 = 12 items
                vec![Z, Z, X, Z, X, Z, X, Z, Z, Z, X, X],
                vec![
                    0u32, 1u32, 2u32, 0u32, 2u32, 1u32, 2u32, 0u32, 1u32, 0u32, 1u32, 2u32,
                ],
                vec![
                    0usize, 3usize, 5usize, 7usize, 9usize, 10usize, 11usize, 12usize,
                ],
            )
        }
        // CCZ: CCZ = exp(-i*(pi/8)*(Z0*Z1*Z2 - Z0*Z2 - Z1*Z2 - Z0*Z1 + Z0 + Z1 + Z2))
        StandardGate::CCZ => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(s, 0.0),  // Z0 Z1 Z2
                    Complex64::new(-s, 0.0), // Z0 Z2
                    Complex64::new(-s, 0.0), // Z1 Z2
                    Complex64::new(-s, 0.0), // Z0 Z1
                    Complex64::new(s, 0.0),  // Z0
                    Complex64::new(s, 0.0),  // Z1
                    Complex64::new(s, 0.0),  // Z2
                ],
                // Term 0: Z(q0) Z(q1) Z(q2)  — 3 items
                // Term 1: Z(q0) Z(q2)          — 2 items
                // Term 2: Z(q1) Z(q2)          — 2 items
                // Term 3: Z(q0) Z(q1)          — 2 items
                // Term 4: Z(q0)                 — 1 item
                // Term 5: Z(q1)                 — 1 item
                // Term 6: Z(q2)                 — 1 item
                // Total: 3+2+2+2+1+1+1 = 12 items
                vec![Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z],
                vec![
                    0u32, 1u32, 2u32, 0u32, 2u32, 1u32, 2u32, 0u32, 1u32, 0u32, 1u32, 2u32,
                ],
                vec![
                    0usize, 3usize, 5usize, 7usize, 9usize, 10usize, 11usize, 12usize,
                ],
            )
        }
        // CSwap (Fredkin): CSwap = exp(-i*(pi/8)*(Z0 - Z0*X1*X2 - Z0*Y1*Y2 - Z0*Z1*Z2 + X1*X2 + Y1*Y2 + Z1*Z2))
        // 7 terms: Z0(-ZXX=ZYY=ZZZ), +XX, +YY, +ZZ
        StandardGate::CSwap => {
            let s = std::f64::consts::PI / 8.0;
            (
                vec![
                    Complex64::new(s, 0.0),  // Z0
                    Complex64::new(-s, 0.0), // Z0 X1 X2
                    Complex64::new(-s, 0.0), // Z0 Y1 Y2
                    Complex64::new(-s, 0.0), // Z0 Z1 Z2
                    Complex64::new(s, 0.0),  // X1 X2
                    Complex64::new(s, 0.0),  // Y1 Y2
                    Complex64::new(s, 0.0),  // Z1 Z2
                ],
                // Term 0: Z(q0)
                // Term 1: Z(q0) X(q1) X(q2)
                // Term 2: Z(q0) Y(q1) Y(q2)
                // Term 3: Z(q0) Z(q1) Z(q2)
                // Term 4: X(q1) X(q2)
                // Term 5: Y(q1) Y(q2)
                // Term 6: Z(q1) Z(q2)
                vec![Z, Z, X, X, Z, Y, Y, Z, Z, Z, X, X, Y, Y, Z, Z],
                vec![
                    0u32, 0u32, 1u32, 2u32, 0u32, 1u32, 2u32, 0u32, 1u32, 2u32, 1u32, 2u32, 1u32,
                    2u32, 1u32, 2u32,
                ],
                vec![
                    0usize, 1usize, 4usize, 7usize, 10usize, 12usize, 14usize, 16usize,
                ],
            )
        }
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
    fn ccx_has_seven_terms() {
        let obs =
            generator_observable(StandardGate::CCX, &[]).expect("CCX should have a generator");
        assert_eq!(obs.num_terms(), 7);
    }
}
