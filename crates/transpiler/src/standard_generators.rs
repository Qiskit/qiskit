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

use num_complex::Complex64;
use qiskit_circuit::operations::StandardGate;

use qiskit_quantum_info::sparse_observable::BitTerm;
use qiskit_quantum_info::sparse_observable::SparseObservable;

/// Return an observable for the generator of `gate`, if we have one.
///
/// `None` means “no special handling, use the generic commutation path”.
pub fn generator_observable(gate: StandardGate) -> Option<SparseObservable> {
    // Helper to create a sorted (indices, bit_terms) pair from a list of (qubit, term)
    fn make_term_data(term_components: &[(u32, BitTerm)]) -> (Vec<u32>, Vec<BitTerm>) {
        let mut sorted_components = term_components.to_vec();
        sorted_components.sort_by_key(|k| k.0);
        sorted_components.into_iter().unzip()
    }

    // Definitions using static slices to avoid generic Vec allocation overhead.
    // Each gate returns (terms_list, num_qubits).
    // terms_list is a slice of terms. Each term is a slice of (qubit, generic Pauli).
    let (definition, num_qubits): (&[&[(u32, BitTerm)]], u32) = match gate {
        // Single Qubit Gates (act on q0)
        // X-type: [X]
        StandardGate::X | StandardGate::RX | StandardGate::SX | StandardGate::SXdg | StandardGate::R => (&[&[(0, BitTerm::X)]], 1),
        StandardGate::Y | StandardGate::RY => (&[&[(0, BitTerm::Y)]], 1),
        StandardGate::Z
        | StandardGate::S
        | StandardGate::Sdg
        | StandardGate::T
        | StandardGate::Tdg
        | StandardGate::RZ
        | StandardGate::Phase
        | StandardGate::U1 => (&[&[(0, BitTerm::Z)]], 1),
        StandardGate::H => (&[&[(0, BitTerm::X)], &[(0, BitTerm::Z)]], 1),

        // Two Qubit Gates (q0, q1)
        // Two Qubit Gates (q0, q1)
        // CX (CNOT): [ZX, IX, ZI] -> Z on 0, X on 1 + locals
        StandardGate::CX | StandardGate::CSX => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::X)],
                &[(1, BitTerm::X)],
                &[(0, BitTerm::Z)],
            ],
            2,
        ),
        // CRX: [ZX, IX]
        StandardGate::CRX => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::X)],
                &[(1, BitTerm::X)],
            ],
            2,
        ),
        // RZX: [ZX] (Pure interaction)
        StandardGate::RZX => (&[&[(0, BitTerm::Z), (1, BitTerm::X)]], 2),
        // CZ, CP, etc: [ZZ, IZ, ZI]
        StandardGate::CZ
        | StandardGate::CPhase
        | StandardGate::CS
        | StandardGate::CSdg
        | StandardGate::CU1 => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::Z)],
                &[(1, BitTerm::Z)],
                &[(0, BitTerm::Z)],
            ],
            2,
        ),
        // CRZ: [ZZ, IZ]
        StandardGate::CRZ => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::Z)],
                &[(1, BitTerm::Z)],
            ],
            2,
        ),
        // RZZ: [ZZ]
        StandardGate::RZZ => (&[&[(0, BitTerm::Z), (1, BitTerm::Z)]], 2),
        // CY: [ZY, IY, ZI]
        StandardGate::CY => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::Y)],
                &[(1, BitTerm::Y)],
                &[(0, BitTerm::Z)],
            ],
            2,
        ),
        // CRY: [ZY, IY]
        StandardGate::CRY => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::Y)],
                &[(1, BitTerm::Y)],
            ],
            2,
        ),
        // Swap: [XX, YY, ZZ]
        StandardGate::Swap => (
            &[
                &[(0, BitTerm::X), (1, BitTerm::X)],
                &[(0, BitTerm::Y), (1, BitTerm::Y)],
                &[(0, BitTerm::Z), (1, BitTerm::Z)],
            ],
            2,
        ),
        // iSwap, XXMinusYY, XXPlusYY: [XX, YY]
        StandardGate::ISwap | StandardGate::XXMinusYY | StandardGate::XXPlusYY => (
            &[
                &[(0, BitTerm::X), (1, BitTerm::X)],
                &[(0, BitTerm::Y), (1, BitTerm::Y)],
            ],
            2,
        ),
        // ECR: [ZX, XI]
        StandardGate::ECR => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::X)], // ZX
                &[(0, BitTerm::X)],                  // XI (X on 0)
            ],
            2,
        ),
        // RXX: [XX]
        StandardGate::RXX => (&[&[(0, BitTerm::X), (1, BitTerm::X)]], 2),
        // RYY: [YY]
        StandardGate::RYY => (&[&[(0, BitTerm::Y), (1, BitTerm::Y)]], 2),
        // CH: [ZX, ZZ]
        StandardGate::CH => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::X)],
                &[(0, BitTerm::Z), (1, BitTerm::Z)],
            ],
            2,
        ),
        // DCX: Not supported due to complex generator structure (non-commuting terms?). 
        // Returning None falls back to generic matrix multiplication check which is safe.
        StandardGate::DCX => return None,

        // Three Qubit Gates (q0, q1, q2) like CCX (Toffoli): [ZZX] (Z on 0, Z on 1, X on 2)
        StandardGate::CCX => (
            &[&[(0, BitTerm::Z), (1, BitTerm::Z), (2, BitTerm::X)]],
            3,
        ),
        // CSwap (Fredkin): sum of [ZXX, ZYY, ZZZ] + [XX, YY, ZZ] + [Z] locals.
        StandardGate::CSwap => (
            &[
                &[(0, BitTerm::Z), (1, BitTerm::X), (2, BitTerm::X)],
                &[(0, BitTerm::Z), (1, BitTerm::Y), (2, BitTerm::Y)],
                &[(0, BitTerm::Z), (1, BitTerm::Z), (2, BitTerm::Z)],
                &[(1, BitTerm::X), (2, BitTerm::X)],
                &[(1, BitTerm::Y), (2, BitTerm::Y)],
                &[(1, BitTerm::Z), (2, BitTerm::Z)],
                &[(0, BitTerm::Z)],
            ],
            3,
        ),
        // CCZ: [ZZZ]
        StandardGate::CCZ => (
            &[&[(0, BitTerm::Z), (1, BitTerm::Z), (2, BitTerm::Z)]],
            3,
        ),

        // Four Qubit Gates like C3X, C3SX, RC3X: [ZZZX]
        StandardGate::C3X | StandardGate::C3SX | StandardGate::RC3X => (
            &[&[
                (0, BitTerm::Z),
                (1, BitTerm::Z),
                (2, BitTerm::Z),
                (3, BitTerm::X),
            ]],
            4,
        ),

        // Global Phase: [] - No generators (identity up to phase) - Empty observable.
        StandardGate::GlobalPhase | StandardGate::I => (&[], 0),

        // Others not handled yet (return None)
        _ => return None,
    };

    let mut all_bit_terms = Vec::new();
    let mut all_indices = Vec::new();
    let mut boundaries = vec![0];
    let mut coeffs = Vec::new();

    for term_pattern in definition {
        let (mut indices, mut bits) = make_term_data(term_pattern);
        all_indices.append(&mut indices);
        all_bit_terms.append(&mut bits);
        boundaries.push(all_bit_terms.len()); // Boundary is length of bit_terms
        coeffs.push(Complex64::new(1.0, 0.0));
    }
    
    // The indices are constructed from fixed patterns which are valid.
    let obs = unsafe {
        SparseObservable::new_unchecked(
            num_qubits,
            coeffs,
            all_bit_terms,
            all_indices,
            boundaries,
        )
    };
    Some(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cx_generator() {
        let obs = generator_observable(StandardGate::CX).expect("CX should have generator");
        assert_eq!(obs.num_qubits(), 2);
        // ZX, IX, ZI
        assert_eq!(obs.num_terms(), 3);
        let bit_terms = obs.bit_terms();
        assert_eq!(bit_terms[0], BitTerm::X); // ZX -> X on 1
        assert_eq!(bit_terms[1], BitTerm::Z); // ZX -> Z on 0
        assert_eq!(bit_terms[2], BitTerm::X); // IX -> X on 1
        assert_eq!(bit_terms[3], BitTerm::Z); // ZI -> Z on 0
    }

    #[test]
    fn test_swap_generator() {
        let obs = generator_observable(StandardGate::Swap).expect("Swap should have generator");
        assert_eq!(obs.num_qubits(), 2);
        // XX + YY + ZZ.
        assert_eq!(obs.num_terms(), 3);
        let bit_terms = obs.bit_terms();
        // Term 1: X, X. Term 2: Y, Y. Term 3: Z, Z.
        assert_eq!(bit_terms[0], BitTerm::X);
        assert_eq!(bit_terms[1], BitTerm::X);
        assert_eq!(bit_terms[2], BitTerm::Y);
        assert_eq!(bit_terms[3], BitTerm::Y);
        assert_eq!(bit_terms[4], BitTerm::Z);
        assert_eq!(bit_terms[5], BitTerm::Z);
    }
    
    #[test]
    fn test_c3x_generator() {
        let obs = generator_observable(StandardGate::C3X).expect("C3X should have generator");
        assert_eq!(obs.num_qubits(), 4);
        assert_eq!(obs.indices(), &[0, 1, 2, 3]);
        assert_eq!(
            obs.bit_terms(),
            &[BitTerm::Z, BitTerm::Z, BitTerm::Z, BitTerm::X]
        );
    }
}
