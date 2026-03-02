// src/sparse_observable/standard_generators.rs

use super::BitTerm;
use super::SparseObservable;
use num_complex::Complex64;
use pyo3::prelude::*;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::StandardGate;

/// Return an observable for the generator of `gate`, if we have one.
#[allow(clippy::result_large_err)]
pub fn generator_observable(
    gate: StandardGate,
    params: &[qiskit_circuit::operations::Param],
) -> Option<SparseObservable> {
    let _params = params;
    let num_qubits = gate.num_qubits();

    use BitTerm::*;

    // Single qubit gates
    if num_qubits == 1 {
        let (coeffs, terms, indices) = match gate {
            StandardGate::H => (
                vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
                vec![X, Z],
                vec![0, 0],
            ),
            StandardGate::X | StandardGate::RX | StandardGate::SX | StandardGate::SXdg => {
                (vec![Complex64::new(1.0, 0.0)], vec![X], vec![0])
            }
            StandardGate::Y | StandardGate::RY => {
                (vec![Complex64::new(1.0, 0.0)], vec![Y], vec![0])
            }
            StandardGate::Z
            | StandardGate::RZ
            | StandardGate::S
            | StandardGate::Sdg
            | StandardGate::T
            | StandardGate::Tdg
            | StandardGate::Phase => (vec![Complex64::new(1.0, 0.0)], vec![Z], vec![0]),
            _ => return None,
        };

        // Construct boundaries for 1-qubit case (trivial: 0, 1, 2... for each term)
        // Actually, if terms has N items, boundaries is satisfy: boundaries[i] maps to start of term i.
        // BitTerms are just terms.
        // For 1 qubit, each term has 1 op.
        // So indices should just be [0, 0, ...] (qubit index 0 for each term).
        // Wait, indices must map term ops to qubits.
        // Yes, vec![0, 0] for H means: Term 0 acts on qubit 0. Term 1 acts on qubit 0.
        // Boundaries: 0, 1, 2...
        // Let's replicate logic from previous implementation.
        let boundaries: Vec<usize> = (0..=terms.len()).collect();

        return Some(
            SparseObservable::new(num_qubits, coeffs, terms, indices, boundaries)
                .expect("invalid 1-qubit generator layout"),
        );
    }

    // Multi-qubit gates
    // The match returns a tuple (coeffs, bit_terms, indices, boundaries) in a flattened
    // Struct-of-Arrays (SoA) layout for efficiency, to avoid allocating many small vectors.
    // Term `i` corresponds to the slice `boundaries[i]..boundaries[i+1]` of `bit_terms` and `indices`.
    let (coeffs, bit_terms, indices, boundaries) = match gate {
        // CX = Z0X1 - Z0 - X1
        // (1, -1, -1) coeffs.
        // Z0 X1 (term 0), Z0 (term 1), X1 (term 2)
        StandardGate::CX | StandardGate::CSX => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            vec![Z, X, Z, X],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CY = Z0Y1 - Z0 - Y1
        StandardGate::CY => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            vec![Z, Y, Z, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CZ = Z0Z1 - Z0 - Z1
        StandardGate::CZ | StandardGate::CS | StandardGate::CSdg => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CRX(t) = -t/4 (Z0X1 - X1)
        StandardGate::CRX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, X, X],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CRY(t) = -t/4 (Z0Y1 - Y1)
        StandardGate::CRY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, Y, Y],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CRZ(t) = -t/4 (Z0Z1 - Z1)
        StandardGate::CRZ => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![Z, Z, Z],
                vec![0, 1, 1],
                vec![0, 2, 3],
            )
        }
        // CPhase(t) = -t/4 (Z0Z1 - Z0 - Z1)
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
                vec![0, 1, 0, 1],
                vec![0, 2, 3, 4],
            )
        }
        // ECR = X0 - Y0X1
        StandardGate::ECR => (
            vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
            vec![X, Y, X], // X0, Y0 X1
            vec![0, 0, 1],
            vec![0, 1, 3],
        ),
        // Swap = X0X1 + Y0Y1 + Z0Z1
        StandardGate::Swap => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            vec![X, X, Y, Y, Z, Z],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 2, 4, 6],
        ),
        // ISwap = X0X1 + Y0Y1
        StandardGate::ISwap => (
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![X, X, Y, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        ),
        // RXX(t) = -t/2 (XX)
        StandardGate::RXX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 2.0, 0.0)],
                vec![X, X],
                vec![0, 1],
                vec![0, 2],
            )
        }
        // RYY(t) = -t/2 (YY)
        StandardGate::RYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 2.0, 0.0)],
                vec![Y, Y],
                vec![0, 1],
                vec![0, 2],
            )
        }
        // RZZ(t) = -t/2 (ZZ)
        StandardGate::RZZ => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 2.0, 0.0)],
                vec![Z, Z],
                vec![0, 1],
                vec![0, 2],
            )
        }
        // RZX(t) = -t/2 (ZX)
        StandardGate::RZX => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta)] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(-t / 2.0, 0.0)],
                vec![Z, X],
                vec![0, 1],
                vec![0, 2],
            )
        }
        // XXPlusYY(t, beta) | XXMinusYY(t, beta)
        StandardGate::XXPlusYY | StandardGate::XXMinusYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta), ..] = _params {
                *theta
            } else {
                1.0
            };
            // Logic for +/- based on variant
            match gate {
                StandardGate::XXPlusYY => (
                    vec![Complex64::new(t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                StandardGate::XXMinusYY => (
                    vec![Complex64::new(t / 4.0, 0.0), Complex64::new(-t / 4.0, 0.0)],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                _ => unreachable!(),
            }
        }
        // CCX = Z0Z1X2 - Z0X2 - Z1X2 + X2
        ),
        // CSwap = Z0 - Z0X1X2 - Z0Y1Y2 - Z0Z1Z2 + X1X2 + Y1Y2 + Z1Z2
        StandardGate::CSwap => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            vec![Z, Z, X, X, Z, Y, Y, Z, Z, Z, X, X, Y, Y, Z, Z],
            vec![0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2],
            vec![0, 1, 4, 7, 10, 12, 14, 16],
        ),
        _ => return None,
    };

    Some(
        SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
            .expect("invalid multi-qubit generator layout"),
    )
}

#[pyfunction(name = "generator_observable")]
#[pyo3(signature = (gate, params = None))]
pub fn generator_observable_py(
    gate: StandardGate,
    params: Option<Vec<qiskit_circuit::operations::Param>>,
) -> Option<SparseObservable> {
    let params = params.unwrap_or_default();
    generator_observable(gate, &params)
}

pub fn standard_generators(m: &Bound<PyModule>) -> PyResult<()> {
    // Re-export StandardGate to ensure it is available and recognized
    m.add_class::<StandardGate>()?;
    m.add_function(wrap_pyfunction!(generator_observable_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::RX, &[]).expect("Rx should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }
}
