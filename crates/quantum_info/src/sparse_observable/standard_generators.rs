// src/sparse_observable/standard_generators.rs

use num_complex::Complex64;
use super::BitTerm;
use super::SparseObservable;

// Temporary: minimal set of standard 1‑qubit gates we care about here.
// This can be wired up to the real gate enum later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardGate {
    Id,
    Rx,
    Ry,
    Rz,
    H,
    S,
    Sdg,
    // TODO: extend if/when we need more.
}

impl StandardGate {
    pub const NUM_VARIANTS: usize = 7;

    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }
}

// StandardGate → Pauli generator(s) as BitTerms.
// Empty slice = “no generator, fall back to the old path”.
static GENERATORS: [&[BitTerm]; StandardGate::NUM_VARIANTS] = {
    use BitTerm::*;
    [
        // Id
        &[],

        // Rx
        &[X],

        // Ry
        &[Y],

        // Rz
        &[Z],

        // H (roughly X+Z in this picture)
        &[X, Z],

        // S
        &[Z],

        // Sdg
        &[Z],
    ]
};

/// Return an observable for the generator of `gate`, if we have one.
///
/// `None` means “no special handling, use the generic commutation path”.
pub fn generator_observable(gate: StandardGate) -> Option<SparseObservable> {
    let idx = gate.as_index();
    let terms = GENERATORS.get(idx)?;
    if terms.is_empty() {
        return None;
    }

    // For now assume a single-qubit generator acting on 1 qubit.
    // You can generalize later if needed.
    let num_qubits = 1;

    // One coefficient per term, all +1 for now.
    let coeffs = vec![Complex64::new(1.0, 0.0); terms.len()];

    // Flatten the BitTerm slice into a Vec<BitTerm>.
    let bit_terms: Vec<BitTerm> = terms.to_vec();

    // Each term uses one BitTerm; indices are 0,1,2,... and boundaries are
    // [0, 1, 2, ..., len].
    let indices: Vec<u32> = (0..bit_terms.len() as u32).collect();
    let boundaries: Vec<usize> = (0..=bit_terms.len()).collect();

    // Safe constructor; unwrap is fine here because we control the layout.
    let obs = SparseObservable::new(
        num_qubits,
        coeffs,
        bit_terms,
        indices,
        boundaries,
    )
    .expect("invalid generator observable layout");

    Some(obs)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::Rx)
            .expect("Rx should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }
}
