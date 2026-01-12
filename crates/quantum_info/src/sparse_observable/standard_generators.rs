use num_complex::Complex64;
use qiskit_circuit::operations::StandardGate;

use super::BitTerm;
use super::SparseObservable;

/// Return an observable for the generator of `gate`, if we have one.
///
/// `None` means “no special handling, use the generic commutation path”.
pub fn generator_observable(gate: StandardGate) -> Option<SparseObservable> {
    let terms: &[BitTerm] = match gate {
        StandardGate::X | StandardGate::SX | StandardGate::SXdg => &[BitTerm::X],
        StandardGate::Y => &[BitTerm::Y],
        StandardGate::Z
        | StandardGate::S
        | StandardGate::Sdg
        | StandardGate::T
        | StandardGate::Tdg => &[BitTerm::Z],
        StandardGate::RX => &[BitTerm::X],
        StandardGate::RY => &[BitTerm::Y],
        StandardGate::RZ | StandardGate::Phase | StandardGate::U1 => &[BitTerm::Z],
        StandardGate::H => &[BitTerm::X, BitTerm::Z],
        _ => return None,
    };

    // For now assume a single-qubit generator acting on 1 qubit.
    let num_qubits = 1;

    // One coefficient per term, all +1 for now.
    let coeffs = vec![Complex64::new(1.0, 0.0); terms.len()];

    // Flatten the BitTerm slice into a Vec<BitTerm>.
    let bit_terms: Vec<BitTerm> = terms.to_vec();

    // Each term uses one BitTerm; for a 1-qubit gate, all terms act on qubit 0.
    let indices: Vec<u32> = vec![0; bit_terms.len()];
    let boundaries: Vec<usize> = (0..=bit_terms.len()).collect();

    // Safe constructor; unwrap is fine here because we control the layout.
    let obs = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
        .expect("invalid generator observable layout");

    Some(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::RX).expect("RX should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }
}
