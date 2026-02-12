// src/sparse_observable/standard_generators.rs

use num_complex::Complex64;
use super::BitTerm;
use super::SparseObservable;
use qiskit_circuit::operations::Param;

// Standard single-qubit gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardGate {
    Id,
    X,
    Y,
    Z,
    Rx,
    Ry,
    Rz,
    H,
    S,
    Sdg,
    SX,
    SXdg,
    T,
    Tdg,
    CX,
    CY,
    CZ,
    CRX,
    CRY,
    CRZ,
    CPhase,
    ECR,
    Swap,
    ISwap,
    RXX,
    RYY,
    RZZ,
    RZX,
    XXPlusYY,
    XXMinusYY,
    CCX,
    CSwap,
    CSX,
    CS,
    CSdg,
}

impl StandardGate {
    pub const NUM_VARIANTS: usize = 22;

    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }

    pub fn num_qubits(self) -> u32 {
        match self {
            Self::Id
            | Self::X
            | Self::Y
            | Self::Z
            | Self::Rx
            | Self::Ry
            | Self::Rz
            | Self::H
            | Self::S
            | Self::Sdg
            | Self::SX
            | Self::SXdg
            | Self::T
            | Self::Tdg => 1,
            Self::CX
            | Self::CY
            | Self::CZ
            | Self::CRX
            | Self::CRY
            | Self::CRZ
            | Self::CPhase
            | Self::ECR
            | Self::Swap
            | Self::ISwap
            | Self::RXX
            | Self::RYY
            | Self::RZZ
            | Self::RZX
            | Self::XXPlusYY
            | Self::XXMinusYY
            | Self::CSX
            | Self::CS
            | Self::CSdg => 2,
            Self::CCX | Self::CSwap => 3,
        }
    }
}

// Mapping of single-qubit gates to their generators.
static SINGLE_QUBIT_GENERATORS: [&[BitTerm]; 14] = {
    use BitTerm::*;
    [
        &[],     // Id
        &[X],    // X
        &[Y],    // Y
        &[Z],    // Z
        &[X],    // Rx
        &[Y],    // Ry
        &[Z],    // Rz
        &[X, Z], // H
        &[Z],    // S
        &[Z],    // Sdg
        &[X],    // SX
        &[X],    // SXdg
        &[Z],    // T
        &[Z],    // Tdg
    ]
};

/// Return an observable for the generator of `gate`, if we have one.
pub fn generator_observable(gate: StandardGate, _params: &[Param]) -> Option<SparseObservable> {
    let num_qubits = gate.num_qubits();

    if num_qubits == 1 {
        let idx = gate.as_index();
        let terms = SINGLE_QUBIT_GENERATORS.get(idx)?;
        if terms.is_empty() {
            return None;
        }
        let coeffs = vec![Complex64::new(1.0, 0.0); terms.len()];
        let bit_terms: Vec<BitTerm> = terms.to_vec();
        let indices: Vec<u32> = (0..bit_terms.len() as u32).map(|_| 0).collect();
        let boundaries: Vec<usize> = (0..=bit_terms.len()).collect();

        return Some(
            SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
                .expect("invalid 1-qubit generator layout"),
        );
    }

    // Multi-qubit gates
    use BitTerm::*;
    let (coeffs, bit_terms, indices, boundaries) = match gate {
        // CX = Z0X1 - Z0 - X1
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
            vec![X, Y, X],
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
        // XXPlusYY(t, beta) = t/4 (XX + YY)
        StandardGate::XXPlusYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta), _] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 4.0, 0.0), Complex64::new(t / 4.0, 0.0)],
                vec![X, X, Y, Y],
                vec![0, 1, 0, 1],
                vec![0, 2, 4],
            )
        }
        // XXMinusYY(t, beta) = t/4 (XX - YY)
        StandardGate::XXMinusYY => {
            let t = if let [qiskit_circuit::operations::Param::Float(theta), _] = _params {
                *theta
            } else {
                1.0
            };
            (
                vec![Complex64::new(t / 4.0, 0.0), Complex64::new(-t / 4.0, 0.0)],
                vec![X, X, Y, Y],
                vec![0, 1, 0, 1],
                vec![0, 2, 4],
            )
        }
        // CCX = Z0Z1X2 - Z0X2 - Z1X2 + X2
        StandardGate::CCX => (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            vec![Z, Z, X, Z, X, Z, X, X],
            vec![0, 1, 2, 0, 2, 1, 2, 2],
            vec![0, 3, 5, 7, 8],
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::Rx, &[])
            .expect("Rx should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }
}
